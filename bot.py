import json
import logging
import os
import sqlite3
import threading
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

# Import TON escrow module (optional - graceful degradation if dependencies missing)
try:
    import ton_escrow
    TON_ESCROW_AVAILABLE = True
except ImportError as e:
    TON_ESCROW_AVAILABLE = False
    logging.warning(f"TON escrow module not available: {e}. Install tonsdk, cryptography, aiohttp.")

# Import auto-poster module
try:
    import auto_poster
    AUTO_POSTER_AVAILABLE = True
except ImportError as e:
    AUTO_POSTER_AVAILABLE = False
    logging.warning(f"Auto-poster module not available: {e}")

# Import notifications module
try:
    import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError as e:
    NOTIFICATIONS_AVAILABLE = False
    logging.warning(f"Notifications module not available: {e}")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE SETUP
# =============================================================================

DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')


def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute('PRAGMA foreign_keys = ON')
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER UNIQUE NOT NULL,
            role TEXT DEFAULT 'user',
            ton_wallet TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Channels table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL,
            telegram_channel_id INTEGER,
            username TEXT NOT NULL,
            name TEXT,
            category TEXT DEFAULT 'general',
            price REAL DEFAULT 0,
            subscribers INTEGER DEFAULT 0,
            avg_views INTEGER DEFAULT 0,
            verified INTEGER DEFAULT 0,
            bot_is_admin INTEGER DEFAULT 0,
            bot_can_post INTEGER DEFAULT 0,
            owner_ton_wallet TEXT,
            verified_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users(id)
        )
    ''')
    
    # Channel admins table (role-based access control)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS channel_admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('owner', 'manager', 'poster')),
            verified_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (channel_id) REFERENCES channels(id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(channel_id, user_id)
        )
    ''')
    
    # Campaigns table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            advertiser_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            text TEXT,
            budget REAL DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (advertiser_id) REFERENCES users(id)
        )
    ''')
    
    # Deals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id INTEGER,
            channel_id INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            escrow_amount REAL DEFAULT 0,
            advertiser_wallet TEXT,
            channel_owner_wallet TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (campaign_id) REFERENCES campaigns(id),
            FOREIGN KEY (channel_id) REFERENCES channels(id)
        )
    ''')
    
    # Escrow wallets table (one per deal)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS escrow_wallets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deal_id INTEGER UNIQUE NOT NULL,
            address TEXT NOT NULL,
            encrypted_private_key TEXT NOT NULL,
            wallet_version TEXT DEFAULT 'v4r2',
            balance REAL DEFAULT 0,
            last_checked TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (deal_id) REFERENCES deals(id)
        )
    ''')
    
    # Escrow transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS escrow_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_id INTEGER NOT NULL,
            tx_hash TEXT UNIQUE,
            tx_type TEXT NOT NULL CHECK(tx_type IN ('deposit', 'release', 'refund')),
            amount REAL NOT NULL,
            from_address TEXT,
            to_address TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (wallet_id) REFERENCES escrow_wallets(id)
        )
    ''')
    
    # Scheduled posts table (one per deal)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scheduled_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deal_id INTEGER UNIQUE NOT NULL,
            channel_id INTEGER NOT NULL,
            ad_text TEXT NOT NULL,
            scheduled_time TIMESTAMP NOT NULL,
            posted_at TIMESTAMP,
            message_id INTEGER,
            hold_hours INTEGER DEFAULT 24,
            release_at TIMESTAMP,
            status TEXT DEFAULT 'scheduled',
            last_verified TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (deal_id) REFERENCES deals(id),
            FOREIGN KEY (channel_id) REFERENCES channels(id)
        )
    ''')
    
    # Migrations for existing databases
    migrations = [
        'ALTER TABLE channels ADD COLUMN telegram_channel_id INTEGER',
        'ALTER TABLE channels ADD COLUMN verified INTEGER DEFAULT 0',
        'ALTER TABLE channels ADD COLUMN bot_is_admin INTEGER DEFAULT 0',
        'ALTER TABLE channels ADD COLUMN bot_can_post INTEGER DEFAULT 0',
        'ALTER TABLE channels ADD COLUMN verified_at TIMESTAMP',
        'ALTER TABLE channels ADD COLUMN owner_ton_wallet TEXT',
        'ALTER TABLE deals ADD COLUMN advertiser_wallet TEXT',
        'ALTER TABLE deals ADD COLUMN channel_owner_wallet TEXT',
        'ALTER TABLE deals ADD COLUMN message_id INTEGER',
        'ALTER TABLE deals ADD COLUMN posted_at TIMESTAMP',
        'ALTER TABLE deals ADD COLUMN hold_hours INTEGER DEFAULT 24',
        'ALTER TABLE users ADD COLUMN ton_wallet TEXT',
    ]
    
    for migration in migrations:
        try:
            cursor.execute(migration)
        except sqlite3.OperationalError:
            pass  # Column already exists
    
    conn.commit()
    conn.close()
    logger.info(f'Database initialized at {DATABASE_PATH}')


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# PERMISSION SYSTEM
# =============================================================================

class ChannelRole:
    """Channel admin roles with permission levels"""
    OWNER = 'owner'      # Full control: accept deals, post ads, release escrow
    MANAGER = 'manager'  # Can accept deals and release escrow
    POSTER = 'poster'    # Can only post ads

    @staticmethod
    def can_accept_deals(role: str) -> bool:
        return role in [ChannelRole.OWNER, ChannelRole.MANAGER]
    
    @staticmethod
    def can_post_ads(role: str) -> bool:
        return role in [ChannelRole.OWNER, ChannelRole.MANAGER, ChannelRole.POSTER]
    
    @staticmethod
    def can_release_escrow(role: str) -> bool:
        return role in [ChannelRole.OWNER, ChannelRole.MANAGER]


async def verify_telegram_admin(bot, telegram_user_id: int, channel_username: str) -> dict:
    """
    Verify if a user is an admin of a Telegram channel via Telegram API.
    Returns dict with 'is_admin', 'can_post', 'can_manage' flags.
    """
    result = {
        'is_admin': False,
        'can_post': False,
        'can_manage': False,
        'telegram_channel_id': None,
        'error': None
    }
    
    try:
        # Get chat info
        chat = await bot.get_chat(channel_username)
        result['telegram_channel_id'] = chat.id
        
        # Get chat member status
        member = await bot.get_chat_member(chat.id, telegram_user_id)
        
        if member.status in ['creator', 'administrator']:
            result['is_admin'] = True
            result['can_post'] = getattr(member, 'can_post_messages', True)
            result['can_manage'] = member.status == 'creator' or getattr(member, 'can_manage_chat', False)
        
        logger.info(f"Verified admin: user={telegram_user_id}, channel={channel_username}, is_admin={result['is_admin']}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error verifying admin: {e}")
    
    return result


def get_user_channel_role(user_id: int, channel_id: int) -> Optional[str]:
    """Get user's role for a specific channel from database"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT role FROM channel_admins 
            WHERE user_id = ? AND channel_id = ?
        ''', (user_id, channel_id))
        row = cursor.fetchone()
        return row['role'] if row else None


def set_channel_admin(channel_id: int, user_id: int, role: str) -> bool:
    """Add or update a channel admin with specified role"""
    if role not in [ChannelRole.OWNER, ChannelRole.MANAGER, ChannelRole.POSTER]:
        return False
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO channel_admins (channel_id, user_id, role, verified_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(channel_id, user_id) DO UPDATE SET 
                role = excluded.role,
                verified_at = CURRENT_TIMESTAMP
        ''', (channel_id, user_id, role))
        conn.commit()
        logger.info(f"Set admin: channel={channel_id}, user={user_id}, role={role}")
        return True


def remove_channel_admin(channel_id: int, user_id: int) -> bool:
    """Remove a channel admin"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM channel_admins WHERE channel_id = ? AND user_id = ?
        ''', (channel_id, user_id))
        conn.commit()
        return cursor.rowcount > 0


def get_channel_admins(channel_id: int) -> List[dict]:
    """Get all admins for a channel"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ca.user_id, ca.role, ca.verified_at, u.telegram_id
            FROM channel_admins ca
            JOIN users u ON ca.user_id = u.id
            WHERE ca.channel_id = ?
        ''', (channel_id,))
        return [dict(row) for row in cursor.fetchall()]


def check_channel_permission(user_id: int, channel_id: int, action: str) -> dict:
    """
    Check if user has permission to perform action on channel.
    Actions: 'accept_deal', 'post_ad', 'release_escrow'
    Returns dict with 'allowed', 'role', 'error'
    """
    result = {'allowed': False, 'role': None, 'error': None}
    
    role = get_user_channel_role(user_id, channel_id)
    if not role:
        result['error'] = 'User is not an admin of this channel'
        return result
    
    result['role'] = role
    
    if action == 'accept_deal':
        result['allowed'] = ChannelRole.can_accept_deals(role)
        if not result['allowed']:
            result['error'] = 'Only owners and managers can accept deals'
    
    elif action == 'post_ad':
        result['allowed'] = ChannelRole.can_post_ads(role)
        if not result['allowed']:
            result['error'] = 'Insufficient permissions to post ads'
    
    elif action == 'release_escrow':
        result['allowed'] = ChannelRole.can_release_escrow(role)
        if not result['allowed']:
            result['error'] = 'Only owners and managers can release escrow'
    
    else:
        result['error'] = f'Unknown action: {action}'
    
    return result


async def verify_and_update_admin(bot, telegram_user_id: int, channel_id: int) -> dict:
    """
    Re-verify admin rights via Telegram API and update database.
    Should be called before every critical action.
    """
    result = {'verified': False, 'role': None, 'error': None}
    
    # Get channel info from database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT username, telegram_channel_id FROM channels WHERE id = ?', (channel_id,))
        channel = cursor.fetchone()
        
        if not channel:
            result['error'] = 'Channel not found'
            return result
        
        # Get user id from telegram_id
        cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_user_id,))
        user = cursor.fetchone()
        if not user:
            result['error'] = 'User not found'
            return result
        
        user_id = user['id']
    
    # Verify via Telegram API
    verification = await verify_telegram_admin(bot, telegram_user_id, channel['username'])
    
    if verification['error']:
        result['error'] = verification['error']
        return result
    
    if not verification['is_admin']:
        # Remove from admins if no longer admin
        remove_channel_admin(channel_id, user_id)
        result['error'] = 'User is no longer an admin of this channel'
        return result
    
    # Determine role based on Telegram permissions
    if verification['can_manage']:
        role = ChannelRole.OWNER
    elif verification['can_post']:
        role = ChannelRole.MANAGER
    else:
        role = ChannelRole.POSTER
    
    # Update database
    set_channel_admin(channel_id, user_id, role)
    
    # Update telegram_channel_id if not set
    if channel['telegram_channel_id'] != verification['telegram_channel_id']:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE channels SET telegram_channel_id = ? WHERE id = ?',
                (verification['telegram_channel_id'], channel_id)
            )
            conn.commit()
    
    result['verified'] = True
    result['role'] = role
    return result


# =============================================================================
# CHANNEL VERIFICATION
# =============================================================================

async def verify_channel(bot, channel_username: str) -> dict:
    """
    Verify a Telegram channel for registration.
    Checks: bot is admin, bot can post, fetches stats.
    Returns verification result dict.
    """
    result = {
        'success': False,
        'verified': False,
        'bot_is_admin': False,
        'bot_can_post': False,
        'telegram_channel_id': None,
        'title': None,
        'subscribers': 0,
        'description': None,
        'error': None
    }
    
    try:
        # Ensure @ prefix
        if not channel_username.startswith('@'):
            channel_username = '@' + channel_username
        
        # Get channel info
        try:
            chat = await bot.get_chat(channel_username)
        except Exception as e:
            error_str = str(e).lower()
            if 'chat not found' in error_str:
                result['error'] = 'Channel not found. Check the username.'
            elif 'bot was kicked' in error_str:
                result['error'] = 'Bot was removed from channel.'
            else:
                result['error'] = f'Cannot access channel: {e}'
            return result
        
        result['telegram_channel_id'] = chat.id
        result['title'] = chat.title
        result['description'] = chat.description
        
        # Get subscriber count (member count for channels)
        try:
            member_count = await bot.get_chat_member_count(chat.id)
            result['subscribers'] = member_count
        except Exception as e:
            logger.warning(f"Could not get subscriber count: {e}")
            result['subscribers'] = 0
        
        # Check if bot is admin
        try:
            bot_member = await bot.get_chat_member(chat.id, bot.id)
            
            if bot_member.status == 'administrator':
                result['bot_is_admin'] = True
                result['bot_can_post'] = getattr(bot_member, 'can_post_messages', False)
                
                if not result['bot_can_post']:
                    result['error'] = 'Bot is admin but cannot post messages. Enable "Post Messages" permission.'
                    return result
                
                result['verified'] = True
                result['success'] = True
                
            elif bot_member.status == 'creator':
                # Bot is owner (unlikely but handle it)
                result['bot_is_admin'] = True
                result['bot_can_post'] = True
                result['verified'] = True
                result['success'] = True
                
            else:
                result['error'] = 'Bot is not an admin of this channel. Add bot as admin with "Post Messages" permission.'
                return result
                
        except Exception as e:
            result['error'] = f'Cannot verify bot status: {e}'
            return result
        
        logger.info(f"Channel verified: {channel_username}, subscribers={result['subscribers']}, can_post={result['bot_can_post']}")
        
    except Exception as e:
        result['error'] = f'Verification failed: {str(e)}'
        logger.error(f"Channel verification error: {e}")
    
    return result


def update_channel_verification(channel_id: int, verification: dict) -> bool:
    """Update channel with verification results"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE channels SET
                    telegram_channel_id = ?,
                    name = COALESCE(?, name),
                    subscribers = ?,
                    verified = ?,
                    bot_is_admin = ?,
                    bot_can_post = ?,
                    verified_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (
                verification.get('telegram_channel_id'),
                verification.get('title'),
                verification.get('subscribers', 0),
                1 if verification.get('verified') else 0,
                1 if verification.get('bot_is_admin') else 0,
                1 if verification.get('bot_can_post') else 0,
                channel_id
            ))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error updating channel verification: {e}")
        return False


async def verify_and_register_channel(bot, channel_username: str, owner_id: int, 
                                       category: str = 'general', price: float = 0) -> dict:
    """
    Verify and register a new channel in one step.
    Returns result with channel data or error.
    """
    result = {'success': False, 'channel': None, 'error': None}
    
    # Verify the channel first
    verification = await verify_channel(bot, channel_username)
    
    if not verification['success']:
        result['error'] = verification['error']
        return result
    
    # Ensure @ prefix
    if not channel_username.startswith('@'):
        channel_username = '@' + channel_username
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Check if channel already exists
            cursor.execute('SELECT id FROM channels WHERE username = ?', (channel_username,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing channel
                cursor.execute('''
                    UPDATE channels SET
                        telegram_channel_id = ?,
                        name = ?,
                        subscribers = ?,
                        verified = 1,
                        bot_is_admin = 1,
                        bot_can_post = 1,
                        verified_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    verification['telegram_channel_id'],
                    verification['title'],
                    verification['subscribers'],
                    existing['id']
                ))
                conn.commit()
                channel_id = existing['id']
                logger.info(f"Updated existing channel {channel_id} - {channel_username}")
            else:
                # Create new channel
                cursor.execute('''
                    INSERT INTO channels (
                        owner_id, telegram_channel_id, username, name, category, price,
                        subscribers, verified, bot_is_admin, bot_can_post, verified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, 1, CURRENT_TIMESTAMP)
                ''', (
                    owner_id,
                    verification['telegram_channel_id'],
                    channel_username,
                    verification['title'],
                    category,
                    price,
                    verification['subscribers']
                ))
                conn.commit()
                channel_id = cursor.lastrowid
                logger.info(f"Created verified channel {channel_id} - {channel_username}")
            
            # Add owner as channel admin
            set_channel_admin(channel_id, owner_id, ChannelRole.OWNER)
            
            result['success'] = True
            result['channel'] = {
                'id': channel_id,
                'owner_id': owner_id,
                'telegram_channel_id': verification['telegram_channel_id'],
                'username': channel_username,
                'name': verification['title'],
                'category': category,
                'price': price,
                'subscribers': verification['subscribers'],
                'verified': True,
                'bot_is_admin': True,
                'bot_can_post': True
            }
            
    except Exception as e:
        result['error'] = f'Database error: {str(e)}'
        logger.error(f"Error registering channel: {e}")
    
    return result


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class EscrowStatus(str, Enum):
    """Escrow states for deals"""
    CREATED = "created"
    REQUESTED = "requested"
    ACCEPTED = "accepted"
    FUNDED = "funded"
    POSTED = "posted"
    VERIFIED = "verified"
    COMPLETED = "completed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


# =============================================================================
# DEAL STATE MACHINE
# =============================================================================

class DealStateMachine:
    """
    Strict state machine for deal transitions.
    Ensures atomic, valid state changes with logging.
    """
    
    # Valid state transitions: current_state -> [allowed_next_states]
    TRANSITIONS = {
        'pending': ['accepted', 'cancelled'],
        'accepted': ['funded', 'cancelled'],
        'funded': ['scheduled', 'posted', 'refunded'],  # Can schedule or post directly
        'scheduled': ['posted', 'cancelled', 'refunded'],  # Awaiting scheduled time
        'posted': ['verified', 'refunded'],
        'verified': ['completed', 'refunded'],
        'completed': [],  # Terminal state
        'refunded': [],   # Terminal state
        'cancelled': [],  # Terminal state
    }
    
    # Human-readable state labels
    STATE_LABELS = {
        'pending': 'Pending Approval',
        'accepted': 'Accepted',
        'funded': 'Escrow Funded',
        'scheduled': 'Post Scheduled',
        'posted': 'Ad Posted',
        'verified': 'Verified',
        'completed': 'Completed',
        'refunded': 'Refunded',
        'cancelled': 'Cancelled'
    }
    
    # Step numbers for timeline UI
    STATE_STEPS = {
        'pending': 1,
        'accepted': 2,
        'funded': 3,
        'posted': 4,
        'verified': 5,
        'completed': 6,
        'refunded': 0,
        'cancelled': 0
    }
    
    @classmethod
    def can_transition(cls, current_state: str, new_state: str) -> bool:
        """Check if transition is valid"""
        allowed = cls.TRANSITIONS.get(current_state, [])
        return new_state in allowed
    
    @classmethod
    def get_allowed_transitions(cls, current_state: str) -> List[str]:
        """Get list of valid next states"""
        return cls.TRANSITIONS.get(current_state, [])
    
    @classmethod
    def is_terminal(cls, state: str) -> bool:
        """Check if state is terminal (no further transitions)"""
        return len(cls.TRANSITIONS.get(state, [])) == 0
    
    @classmethod
    def get_step(cls, state: str) -> int:
        """Get step number for timeline display"""
        return cls.STATE_STEPS.get(state, 1)
    
    @classmethod
    def get_label(cls, state: str) -> str:
        """Get human-readable label"""
        return cls.STATE_LABELS.get(state, state.title())


def transition_deal_state(deal_id: int, new_state: str, actor_telegram_id: int = None) -> dict:
    """
    Atomically transition deal to new state.
    Validates transition, updates DB, logs change.
    
    Returns dict with 'success', 'deal', 'error', 'old_state', 'new_state'
    """
    result = {
        'success': False,
        'deal': None,
        'error': None,
        'old_state': None,
        'new_state': None
    }
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get current deal state (with row lock via transaction)
            cursor.execute('SELECT * FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()
            
            if not deal:
                result['error'] = 'Deal not found'
                return result
            
            current_state = deal['status']
            result['old_state'] = current_state
            
            # Check if transition is valid
            if not DealStateMachine.can_transition(current_state, new_state):
                allowed = DealStateMachine.get_allowed_transitions(current_state)
                result['error'] = f"Invalid transition: {current_state} → {new_state}. Allowed: {allowed}"
                return result
            
            # Perform atomic update
            cursor.execute('''
                UPDATE deals SET status = ? WHERE id = ? AND status = ?
            ''', (new_state, deal_id, current_state))
            
            if cursor.rowcount == 0:
                result['error'] = 'State changed by another process (concurrent modification)'
                return result
            
            conn.commit()
            
            # Log the transition
            log_entry = {
                'deal_id': deal_id,
                'old_state': current_state,
                'new_state': new_state,
                'actor_telegram_id': actor_telegram_id,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Deal state transition: {log_entry}")
            
            # Fetch updated deal
            cursor.execute('''
                SELECT d.*, c.username as channel_handle, camp.title as campaign_title
                FROM deals d
                LEFT JOIN channels c ON d.channel_id = c.id
                LEFT JOIN campaigns camp ON d.campaign_id = camp.id
                WHERE d.id = ?
            ''', (deal_id,))
            updated_deal = cursor.fetchone()
            
            result['success'] = True
            result['new_state'] = new_state
            result['deal'] = {
                'id': updated_deal['id'],
                'campaign_id': updated_deal['campaign_id'],
                'channel_id': updated_deal['channel_id'],
                'status': updated_deal['status'],
                'escrow_amount': updated_deal['escrow_amount'],
                'channel': updated_deal['channel_handle'],
                'title': updated_deal['campaign_title'],
                'step': DealStateMachine.get_step(new_state),
                'label': DealStateMachine.get_label(new_state),
                'is_terminal': DealStateMachine.is_terminal(new_state),
                'allowed_transitions': DealStateMachine.get_allowed_transitions(new_state)
            }
            
    except Exception as e:
        result['error'] = f'Database error: {str(e)}'
        logger.error(f"Deal transition error: {e}")
    
    return result


async def send_deal_notification(bot, deal_id: int, event_type: str, extra_data: dict = None):
    """
    Send notification for a deal event.
    
    Args:
        bot: Telegram bot instance
        deal_id: Deal ID
        event_type: Event type (accepted, funded, posted, completed, refunded, etc.)
        extra_data: Additional template variables (optional)
    """
    if not NOTIFICATIONS_AVAILABLE:
        logger.debug("Notifications module not available, skipping notification")
        return
    
    try:
        # Get deal data and participant IDs
        deal_data = notifications.get_deal_data_for_notification(deal_id)
        if not deal_data:
            logger.warning(f"Could not get deal data for notification: deal_id={deal_id}")
            return
        
        # Merge extra data if provided
        if extra_data:
            deal_data.update(extra_data)
        
        # Send notifications to appropriate participants
        result = await notifications.notify_deal_participants(
            bot=bot,
            event_type=event_type,
            data=deal_data,
            advertiser_telegram_id=deal_data.get('advertiser_telegram_id'),
            channel_owner_telegram_id=deal_data.get('channel_owner_telegram_id')
        )
        
        if result['notifications_sent'] > 0:
            logger.info(f"Sent {result['notifications_sent']} notification(s) for deal {deal_id} ({event_type})")
        
        if result['errors']:
            for err in result['errors']:
                logger.warning(f"Notification error: {err}")
                
    except Exception as e:
        logger.error(f"Error sending deal notification: {e}")


def get_deal_with_state_info(deal_id: int) -> dict:
    """Get deal with full state machine info"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.*, c.username as channel_handle, c.name as channel_name,
                       camp.title as campaign_title
                FROM deals d
                LEFT JOIN channels c ON d.channel_id = c.id
                LEFT JOIN campaigns camp ON d.campaign_id = camp.id
                WHERE d.id = ?
            ''', (deal_id,))
            deal = cursor.fetchone()
            
            if not deal:
                return None
            
            state = deal['status']
            return {
                'id': deal['id'],
                'campaign_id': deal['campaign_id'],
                'channel_id': deal['channel_id'],
                'channel': deal['channel_handle'],
                'channel_name': deal['channel_name'],
                'title': deal['campaign_title'] or f"Deal #{deal['id']}",
                'status': state,
                'label': DealStateMachine.get_label(state),
                'step': DealStateMachine.get_step(state),
                'escrow_amount': deal['escrow_amount'],
                'is_terminal': DealStateMachine.is_terminal(state),
                'allowed_transitions': DealStateMachine.get_allowed_transitions(state),
                'created_at': deal['created_at']
            }
    except Exception as e:
        logger.error(f"Error getting deal: {e}")
        return None


class WebAppAction(str, Enum):
    """Available actions from Web App"""
    CREATE_CAMPAIGN = "create_campaign"
    ADD_CHANNEL = "add_channel"
    SELECT_CHANNELS = "select_channels"
    VIEW_MARKETPLACE = "view_marketplace"
    MANAGE_CAMPAIGNS = "manage_campaigns"
    MANAGE_CHANNELS = "manage_channels"


@dataclass
class Campaign:
    """Advertising campaign data structure"""
    id: str
    advertiser_id: int
    title: str
    description: str
    budget: float
    category: str = "general"
    target_language: str = "en"
    min_subscribers: int = 1000
    expected_views_min: int = 500
    expected_views_max: int = 10000
    status: str = "pending"
    escrow_status: str = EscrowStatus.CREATED
    selected_channels: List[str] = field(default_factory=list)
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class ChannelListing:
    """Channel listing data structure with enhanced fields"""
    id: str
    publisher_id: int
    channel_handle: str
    channel_name: str
    category: str
    subscribers: int
    avg_views: int
    price_per_post: float
    language: str = "en"
    status: str = "active"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class Deal:
    """Deal between advertiser and channel owner"""
    id: str
    campaign_id: str
    channel_id: str
    advertiser_id: int
    publisher_id: int
    amount: float
    escrow_status: str = EscrowStatus.CREATED
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


# =============================================================================
# MOCK DATA FOR MVP
# =============================================================================

def get_mock_channels() -> List[dict]:
    """Return mock channel listings for MVP demo"""
    return [
        {
            "id": "ch1",
            "channel_handle": "@cryptonews_hub",
            "channel_name": "Crypto News Hub",
            "category": "crypto",
            "subscribers": 45000,
            "avg_views": 8500,
            "price_per_post": 50,
            "language": "en"
        },
        {
            "id": "ch2",
            "channel_handle": "@finance_daily",
            "channel_name": "Finance Daily",
            "category": "finance",
            "subscribers": 32000,
            "avg_views": 5200,
            "price_per_post": 35,
            "language": "en"
        },
        {
            "id": "ch3",
            "channel_handle": "@nft_world",
            "channel_name": "NFT World",
            "category": "nft",
            "subscribers": 28000,
            "avg_views": 6100,
            "price_per_post": 45,
            "language": "en"
        },
        {
            "id": "ch4",
            "channel_handle": "@gaming_zone",
            "channel_name": "Gaming Zone",
            "category": "gaming",
            "subscribers": 85000,
            "avg_views": 15000,
            "price_per_post": 80,
            "language": "en"
        },
        {
            "id": "ch5",
            "channel_handle": "@defi_insider",
            "channel_name": "DeFi Insider",
            "category": "crypto",
            "subscribers": 18000,
            "avg_views": 3200,
            "price_per_post": 25,
            "language": "en"
        },
        {
            "id": "ch6",
            "channel_handle": "@tech_pulse",
            "channel_name": "Tech Pulse",
            "category": "tech",
            "subscribers": 52000,
            "avg_views": 9800,
            "price_per_post": 55,
            "language": "en"
        },
        {
            "id": "ch7",
            "channel_handle": "@blockchain_now",
            "channel_name": "Blockchain Now",
            "category": "crypto",
            "subscribers": 38000,
            "avg_views": 7200,
            "price_per_post": 42,
            "language": "en"
        }
    ]


# =============================================================================
# BOT CLASS
# =============================================================================

class AdEscrowBot:
    """Main bot class for TG AdEscrow"""
    
    def __init__(self, token: str):
        """Initialize the bot"""
        self.token = token
        self.application = Application.builder().token(token).build()
        self.app = self.application  # Alias for auto_poster & API endpoints
        
        # In-memory storage (replace with database in production)
        self.campaigns: Dict[str, Campaign] = {}
        self.channels: Dict[str, ChannelListing] = {}
        self.deals: Dict[str, Deal] = {}
        self.user_sessions: Dict[int, Dict[str, Any]] = {}
        
        # Load mock channels
        self._load_mock_channels()
        
        self._setup_handlers()
    
    def _load_mock_channels(self):
        """Load mock channel data for MVP"""
        for ch_data in get_mock_channels():
            channel = ChannelListing(
                id=ch_data["id"],
                publisher_id=0,  # Mock publisher
                channel_handle=ch_data["channel_handle"],
                channel_name=ch_data["channel_name"],
                category=ch_data["category"],
                subscribers=ch_data["subscribers"],
                avg_views=ch_data["avg_views"],
                price_per_post=ch_data["price_per_post"],
                language=ch_data["language"]
            )
            self.channels[channel.id] = channel
        logger.info(f"Loaded {len(self.channels)} mock channels")
    
    def _setup_handlers(self):
        """Configure all bot handlers"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("menu", self.menu_command))
        
        self.application.add_handler(
            MessageHandler(filters.StatusUpdate.WEB_APP_DATA, self.handle_webapp_data)
        )
        
        self.application.add_error_handler(self.error_handler)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command with Mini App button"""
        welcome_text = (
            "🤖 *Welcome to TG AdEscrow Bot!*\n\n"
            "Your trusted marketplace for Telegram advertising with escrow protection.\n\n"
            "🔐 *Secure Transactions*\n"
            "💰 *Transparent Pricing*\n"
            "📊 *Real-time Analytics*\n\n"
            "Click the button below to open the Mini App and get started!"
        )
        
        # Get WebApp URL from environment or use Flask server URL
        webapp_url = os.getenv("WEBAPP_URL", "")
        if not webapp_url:
            # Default to the Koyeb deployment URL
            webapp_url = os.getenv("KOYEB_PUBLIC_DOMAIN", "")
            if webapp_url and not webapp_url.startswith("http"):
                webapp_url = f"https://{webapp_url}"
        
        keyboard = [
            [InlineKeyboardButton(
                text="🚀 Open Ad Marketplace",
                web_app=WebAppInfo(url=webapp_url) if webapp_url else None
            )] if webapp_url else [],
            [InlineKeyboardButton("📋 Help Guide", callback_data="help")]
        ]
        # Filter out empty rows
        keyboard = [row for row in keyboard if row]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show main menu with Mini App button"""
        menu_text = (
            "📱 *TG AdEscrow Main Menu*\n\n"
            "Select an option:\n"
            "• 📢 Create Advertising Campaign\n"
            "• 📺 List Your Channel\n"
            "• 🔍 Browse Marketplace\n"
            "• 📊 View Deals Status\n\n"
            "Open the Mini App for the full experience!"
        )
        
        webapp_url = os.getenv("WEBAPP_URL", "")
        if not webapp_url:
            webapp_url = os.getenv("KOYEB_PUBLIC_DOMAIN", "")
            if webapp_url and not webapp_url.startswith("http"):
                webapp_url = f"https://{webapp_url}"
        
        keyboard = []
        if webapp_url:
            keyboard.append([InlineKeyboardButton(
                text="📱 Open Marketplace",
                web_app=WebAppInfo(url=webapp_url)
            )])
        keyboard.append([
            InlineKeyboardButton("ℹ️ Help", callback_data="help"),
            InlineKeyboardButton("📞 Support", callback_data="support")
        ])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            menu_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show help information"""
        help_text = (
            "🆘 *TG AdEscrow Help*\n\n"
            "*For Advertisers:*\n"
            "1. Open the Mini App\n"
            "2. Create a campaign with budget\n"
            "3. Select target channels\n"
            "4. Fund escrow and wait for posting\n\n"
            "*For Channel Owners:*\n"
            "1. List your channel with stats\n"
            "2. Set your price per post\n"
            "3. Accept campaign requests\n"
            "4. Post and get paid!\n\n"
            "*Commands:*\n"
            "/start - Welcome message\n"
            "/menu - Open main menu\n"
            "/help - This help message"
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def handle_webapp_data(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process data received from Telegram Mini App"""
        try:
            web_app_data = update.message.web_app_data
            data_str = web_app_data.data
            
            logger.info(f"Received Web App data from user {update.effective_user.id}")
            
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from Web App: {e}")
                await update.message.reply_text(
                    "❌ *Error processing data*\n\nPlease try again.",
                    parse_mode='Markdown'
                )
                return
            
            action = data.get('action', '')
            user_id = update.effective_user.id
            
            # Process action
            if action == 'create_campaign':
                response = self._handle_campaign_creation(data, user_id)
            elif action == 'add_channel':
                response = self._handle_channel_registration(data, user_id)
            elif action == 'select_channels':
                response = self._handle_channel_selection(data, user_id)
            else:
                response = f"✅ Received: {action}"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error processing Web App data: {e}", exc_info=True)
            await update.message.reply_text(
                "⚠️ *An error occurred*\n\nPlease try again.",
                parse_mode='Markdown'
            )
    
    def _handle_campaign_creation(self, data: dict, user_id: int) -> str:
        """Handle campaign creation"""
        campaign_id = f"camp_{user_id}_{int(datetime.now().timestamp())}"
        campaign = Campaign(
            id=campaign_id,
            advertiser_id=user_id,
            title=data.get('title', 'Untitled'),
            description=data.get('description', ''),
            budget=float(data.get('budget', 0)),
            category=data.get('category', 'general'),
            target_language=data.get('language', 'en'),
            min_subscribers=int(data.get('min_subscribers', 1000)),
            expected_views_min=int(data.get('views_min', 500)),
            expected_views_max=int(data.get('views_max', 10000))
        )
        self.campaigns[campaign_id] = campaign
        
        return (
            f"✅ *Campaign Created!*\n\n"
            f"*Title:* {campaign.title}\n"
            f"*Budget:* {campaign.budget} TON\n"
            f"*ID:* `{campaign_id}`\n\n"
            "Now select channels in the Mini App!"
        )
    
    def _handle_channel_registration(self, data: dict, user_id: int) -> str:
        """Handle channel registration"""
        channel_id = f"chan_{user_id}_{int(datetime.now().timestamp())}"
        channel = ChannelListing(
            id=channel_id,
            publisher_id=user_id,
            channel_handle=data.get('channel_handle', ''),
            channel_name=data.get('channel_name', ''),
            category=data.get('category', 'general'),
            subscribers=int(data.get('subscribers', 0)),
            avg_views=int(data.get('avg_views', 0)),
            price_per_post=float(data.get('price_per_post', 0)),
            language=data.get('language', 'en')
        )
        self.channels[channel_id] = channel
        
        return (
            f"✅ *Channel Registered!*\n\n"
            f"*Channel:* {channel.channel_handle}\n"
            f"*Price:* {channel.price_per_post} TON/post\n"
            f"*ID:* `{channel_id}`"
        )
    
    def _handle_channel_selection(self, data: dict, user_id: int) -> str:
        """Handle advertiser selecting channels"""
        campaign_id = data.get('campaign_id', '')
        selected_channels = data.get('channels', [])
        
        if campaign_id in self.campaigns:
            campaign = self.campaigns[campaign_id]
            campaign.selected_channels = selected_channels
            campaign.escrow_status = EscrowStatus.FUNDED
            
            # Create deals for each selected channel
            for ch_id in selected_channels:
                if ch_id in self.channels:
                    channel = self.channels[ch_id]
                    deal_id = f"deal_{int(datetime.now().timestamp())}_{ch_id[:8]}"
                    deal = Deal(
                        id=deal_id,
                        campaign_id=campaign_id,
                        channel_id=ch_id,
                        advertiser_id=user_id,
                        publisher_id=channel.publisher_id,
                        amount=channel.price_per_post,
                        escrow_status=EscrowStatus.FUNDED
                    )
                    self.deals[deal_id] = deal
            
            return (
                f"✅ *Channels Selected!*\n\n"
                f"*Campaign:* {campaign.title}\n"
                f"*Channels:* {len(selected_channels)}\n"
                f"*Escrow Status:* FUNDED\n\n"
                "Waiting for channel owners to post..."
            )
        
        return "❌ Campaign not found"
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors"""
        logger.error(f"Error: {context.error}", exc_info=context.error)
        try:
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "⚠️ An error occurred. Please try again.",
                    parse_mode='Markdown'
                )
        except Exception:
            pass
    
    def run(self):
        """Start the bot"""
        logger.info("Starting TG AdEscrow Bot...")
        self.application.run_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )


# =============================================================================
# FLASK APP AND API ENDPOINTS (SQLite-backed)
# =============================================================================

flask_app = Flask(__name__, static_folder='miniapp')
bot_instance = None


@flask_app.route('/')
def serve_miniapp():
    """Serve the Mini App index.html"""
    return send_from_directory(flask_app.static_folder, 'index.html')


@flask_app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(flask_app.static_folder, filename)


# -----------------------------------------------------------------------------
# AUTH API
# -----------------------------------------------------------------------------

@flask_app.route('/api/auth', methods=['POST'])
def api_auth():
    """Auto-register user by telegram_id"""
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        
        if not telegram_id:
            return jsonify({'success': False, 'error': 'telegram_id is required'}), 400
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('SELECT * FROM users WHERE telegram_id = ?', (telegram_id,))
            row = cursor.fetchone()
            
            if row:
                user = dict(row)
            else:
                # Auto-register new user
                cursor.execute(
                    'INSERT INTO users (telegram_id, role) VALUES (?, ?)',
                    (telegram_id, 'user')
                )
                conn.commit()
                user = {
                    'id': cursor.lastrowid,
                    'telegram_id': telegram_id,
                    'role': 'user'
                }
                logger.info(f"API: Auto-registered user {telegram_id}")
            
            return jsonify({'success': True, 'user': user})
            
    except Exception as e:
        logger.error(f"Error in auth: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# -----------------------------------------------------------------------------
# CHANNELS API
# -----------------------------------------------------------------------------

@flask_app.route('/api/channels', methods=['GET'])
def api_get_channels():
    """Get all channels from database"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, owner_id, username, name, category, price, subscribers, avg_views, created_at
                FROM channels ORDER BY subscribers DESC
            ''')
            rows = cursor.fetchall()
            
            channels = []
            for row in rows:
                channels.append({
                    'id': row['id'],
                    'owner_id': row['owner_id'],
                    'handle': row['username'],
                    'name': row['name'] or row['username'],
                    'category': row['category'],
                    'price': row['price'],
                    'subscribers': row['subscribers'],
                    'views': row['avg_views'],
                    'created_at': row['created_at']
                })
            
            return jsonify({'success': True, 'channels': channels})
            
    except Exception as e:
        logger.error(f"Error getting channels: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/channels', methods=['POST'])
def api_create_channel():
    """Register a new channel"""
    try:
        data = request.get_json() or {}
        
        owner_id = data.get('owner_id')
        username = data.get('username') or data.get('channel_handle', '')
        name = data.get('name') or data.get('channel_name', '')
        category = data.get('category', 'general')
        price = float(data.get('price') or data.get('price_per_post', 0))
        subscribers = int(data.get('subscribers', 0))
        avg_views = int(data.get('avg_views') or data.get('views', 0))
        
        if not username:
            return jsonify({'success': False, 'error': 'username is required'}), 400
        
        # Ensure @ prefix
        if not username.startswith('@'):
            username = '@' + username
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Ensure owner exists (auto-create if telegram_id provided)
            if not owner_id and data.get('user_id'):
                # Treat user_id as telegram_id and get/create user
                telegram_id = data.get('user_id')
                cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
                row = cursor.fetchone()
                if row:
                    owner_id = row['id']
                else:
                    cursor.execute('INSERT INTO users (telegram_id) VALUES (?)', (telegram_id,))
                    owner_id = cursor.lastrowid
            
            cursor.execute('''
                INSERT INTO channels (owner_id, username, name, category, price, subscribers, avg_views)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (owner_id or 0, username, name, category, price, subscribers, avg_views))
            conn.commit()
            
            channel_id = cursor.lastrowid
            logger.info(f"API: Created channel {channel_id} - {username}")
            
            return jsonify({
                'success': True,
                'channel': {
                    'id': channel_id,
                    'owner_id': owner_id,
                    'username': username,
                    'name': name,
                    'category': category,
                    'price': price,
                    'subscribers': subscribers,
                    'avg_views': avg_views
                }
            })
            
    except Exception as e:
        logger.error(f"Error creating channel: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# -----------------------------------------------------------------------------
# CAMPAIGNS API
# -----------------------------------------------------------------------------

@flask_app.route('/api/campaign/create', methods=['POST'])
def api_create_campaign():
    """Create a new advertising campaign"""
    try:
        data = request.get_json() or {}
        
        advertiser_id = data.get('advertiser_id')
        title = data.get('title', 'Untitled Campaign')
        text = data.get('text') or data.get('description', '')
        budget = float(data.get('budget', 0))
        
        if not title:
            return jsonify({'success': False, 'error': 'title is required'}), 400
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Handle user_id as telegram_id
            if not advertiser_id and data.get('user_id'):
                telegram_id = data.get('user_id')
                cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
                row = cursor.fetchone()
                if row:
                    advertiser_id = row['id']
                else:
                    cursor.execute('INSERT INTO users (telegram_id) VALUES (?)', (telegram_id,))
                    advertiser_id = cursor.lastrowid
            
            cursor.execute('''
                INSERT INTO campaigns (advertiser_id, title, text, budget, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (advertiser_id or 0, title, text, budget, 'pending'))
            conn.commit()
            
            campaign_id = cursor.lastrowid
            logger.info(f"API: Created campaign {campaign_id} - {title}")
            
            return jsonify({
                'success': True,
                'campaign': {
                    'id': campaign_id,
                    'advertiser_id': advertiser_id,
                    'title': title,
                    'text': text,
                    'budget': budget,
                    'status': 'pending'
                }
            })
            
    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# -----------------------------------------------------------------------------
# DEALS API
# -----------------------------------------------------------------------------

@flask_app.route('/api/deals', methods=['GET'])
def api_get_deals():
    """Get all deals with state machine info"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.id, d.campaign_id, d.channel_id, d.status, d.escrow_amount, d.created_at,
                       c.username as channel_handle, c.name as channel_name,
                       camp.title as campaign_title
                FROM deals d
                LEFT JOIN channels c ON d.channel_id = c.id
                LEFT JOIN campaigns camp ON d.campaign_id = camp.id
                ORDER BY d.created_at DESC
            ''')
            rows = cursor.fetchall()
            
            deals = []
            for row in rows:
                state = row['status']
                deals.append({
                    'id': row['id'],
                    'campaign_id': row['campaign_id'],
                    'channel_id': row['channel_id'],
                    'status': state,
                    'label': DealStateMachine.get_label(state),
                    'escrow_amount': row['escrow_amount'],
                    'amount': row['escrow_amount'],
                    'title': row['campaign_title'] or row['channel_name'] or f"Deal #{row['id']}",
                    'channel': row['channel_handle'],
                    'type': 'deal',
                    'step': DealStateMachine.get_step(state),
                    'is_terminal': DealStateMachine.is_terminal(state),
                    'allowed_transitions': DealStateMachine.get_allowed_transitions(state),
                    'created_at': row['created_at']
                })
            
            return jsonify({'success': True, 'deals': deals})
            
    except Exception as e:
        logger.error(f"Error getting deals: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>', methods=['GET'])
def api_get_single_deal(deal_id):
    """Get single deal with full state info"""
    deal = get_deal_with_state_info(deal_id)
    if deal:
        return jsonify({'success': True, 'deal': deal})
    return jsonify({'success': False, 'error': 'Deal not found'}), 404


@flask_app.route('/api/deals', methods=['POST'])
def api_create_deal_via_deals():
    """Create a deal (POST to /api/deals for backwards compatibility)"""
    return api_create_deal()


@flask_app.route('/api/deal/create', methods=['POST'])
def api_create_deal():
    """Create a new deal"""
    try:
        data = request.get_json() or {}
        
        campaign_id = data.get('campaign_id')
        channel_id = data.get('channel_id')
        escrow_amount = float(data.get('escrow_amount', 0))
        status = data.get('status', 'pending')
        
        if not channel_id:
            return jsonify({'success': False, 'error': 'channel_id is required'}), 400
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO deals (campaign_id, channel_id, status, escrow_amount)
                VALUES (?, ?, ?, ?)
            ''', (campaign_id, channel_id, status, escrow_amount))
            conn.commit()
            
            deal_id = cursor.lastrowid
            logger.info(f"API: Created deal {deal_id}")
            
            return jsonify({
                'success': True,
                'deal': {
                    'id': deal_id,
                    'campaign_id': campaign_id,
                    'channel_id': channel_id,
                    'status': status,
                    'escrow_amount': escrow_amount
                }
            })
            
    except Exception as e:
        logger.error(f"Error creating deal: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/status', methods=['POST'])
def api_update_deal_status(deal_id):
    """Update deal status using state machine with strict transitions"""
    try:
        data = request.get_json() or {}
        new_status = data.get('status', '')
        telegram_id = data.get('telegram_id')
        
        if not new_status:
            return jsonify({'success': False, 'error': 'status is required'}), 400
        
        # Use state machine for transition
        result = transition_deal_state(deal_id, new_status, telegram_id)
        
        if result['success']:
            return jsonify({
                'success': True,
                'deal': result['deal'],
                'old_status': result['old_state'],
                'new_status': result['new_state']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'current_status': result['old_state']
            }), 400
            
    except Exception as e:
        logger.error(f"Error updating deal: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/transition', methods=['POST'])
def api_transition_deal(deal_id):
    """Transition deal to new state with full validation"""
    try:
        data = request.get_json() or {}
        new_state = data.get('state') or data.get('status')
        telegram_id = data.get('telegram_id')
        
        if not new_state:
            return jsonify({'success': False, 'error': 'state is required'}), 400
        
        # Check what transitions are allowed
        deal = get_deal_with_state_info(deal_id)
        if not deal:
            return jsonify({'success': False, 'error': 'Deal not found'}), 404
        
        if new_state not in deal['allowed_transitions']:
            return jsonify({
                'success': False,
                'error': f"Cannot transition from '{deal['status']}' to '{new_state}'",
                'current_status': deal['status'],
                'allowed_transitions': deal['allowed_transitions']
            }), 400
        
        # Perform transition
        result = transition_deal_state(deal_id, new_state, telegram_id)
        
        if result['success']:
            # Send notification for this state change
            if bot_instance and bot_instance.application:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        send_deal_notification(
                            bot_instance.application.bot,
                            deal_id,
                            new_state
                        )
                    )
                    loop.close()
                except Exception as notif_err:
                    logger.warning(f"Notification error: {notif_err}")
            
            return jsonify({
                'success': True,
                'deal': result['deal'],
                'transition': f"{result['old_state']} → {result['new_state']}"
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
            
    except Exception as e:
        logger.error(f"Error transitioning deal: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# -----------------------------------------------------------------------------
# PERMISSION-PROTECTED DEAL ACTIONS
# -----------------------------------------------------------------------------

@flask_app.route('/api/deal/<int:deal_id>/accept', methods=['POST'])
def api_accept_deal(deal_id):
    """Accept a deal - requires owner or manager role"""
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        
        if not telegram_id:
            return jsonify({'success': False, 'error': 'telegram_id is required'}), 400
        
        # Get deal and channel info
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT channel_id FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()
            
            if not deal:
                return jsonify({'success': False, 'error': 'Deal not found'}), 404
            
            # Get user_id from telegram_id
            cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
            user = cursor.fetchone()
            if not user:
                return jsonify({'success': False, 'error': 'User not found'}), 404
            
            user_id = user['id']
            channel_id = deal['channel_id']
        
        # Check permission
        permission = check_channel_permission(user_id, channel_id, 'accept_deal')
        
        if not permission['allowed']:
            return jsonify({
                'success': False, 
                'error': permission['error'],
                'role': permission['role']
            }), 403
        
        # Update deal status
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE deals SET status = 'accepted' WHERE id = ?", (deal_id,))
            conn.commit()
        
        logger.info(f"Deal {deal_id} accepted by user {user_id} (role: {permission['role']})")
        
        return jsonify({
            'success': True,
            'deal_id': deal_id,
            'status': 'accepted',
            'role': permission['role']
        })
        
    except Exception as e:
        logger.error(f"Error accepting deal: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/post', methods=['POST'])
def api_post_ad(deal_id):
    """Mark deal as posted - requires poster role or higher"""
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        
        if not telegram_id:
            return jsonify({'success': False, 'error': 'telegram_id is required'}), 400
        
        # Get deal info
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT channel_id, status FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()
            
            if not deal:
                return jsonify({'success': False, 'error': 'Deal not found'}), 404
            
            if deal['status'] not in ['accepted', 'funded', 'escrow']:
                return jsonify({'success': False, 'error': 'Deal must be accepted or funded before posting'}), 400
            
            cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
            user = cursor.fetchone()
            if not user:
                return jsonify({'success': False, 'error': 'User not found'}), 404
            
            user_id = user['id']
            channel_id = deal['channel_id']
        
        # Check permission
        permission = check_channel_permission(user_id, channel_id, 'post_ad')
        
        if not permission['allowed']:
            return jsonify({
                'success': False,
                'error': permission['error'],
                'role': permission['role']
            }), 403
        
        # Update deal status
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE deals SET status = 'posted' WHERE id = ?", (deal_id,))
            conn.commit()
        
        logger.info(f"Deal {deal_id} marked as posted by user {user_id} (role: {permission['role']})")
        
        return jsonify({
            'success': True,
            'deal_id': deal_id,
            'status': 'posted',
            'role': permission['role']
        })
        
    except Exception as e:
        logger.error(f"Error posting ad: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/release', methods=['POST'])
def api_release_escrow(deal_id):
    """Release escrow - requires owner or manager role"""
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        
        if not telegram_id:
            return jsonify({'success': False, 'error': 'telegram_id is required'}), 400
        
        # Get deal info
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT channel_id, status FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()
            
            if not deal:
                return jsonify({'success': False, 'error': 'Deal not found'}), 404
            
            if deal['status'] not in ['posted', 'verified']:
                return jsonify({'success': False, 'error': 'Deal must be posted or verified before escrow release'}), 400
            
            cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
            user = cursor.fetchone()
            if not user:
                return jsonify({'success': False, 'error': 'User not found'}), 404
            
            user_id = user['id']
            channel_id = deal['channel_id']
        
        # Check permission
        permission = check_channel_permission(user_id, channel_id, 'release_escrow')
        
        if not permission['allowed']:
            return jsonify({
                'success': False,
                'error': permission['error'],
                'role': permission['role']
            }), 403
        
        # Update deal status
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE deals SET status = 'completed' WHERE id = ?", (deal_id,))
            conn.commit()
        
        logger.info(f"Deal {deal_id} escrow released by user {user_id} (role: {permission['role']})")
        
        return jsonify({
            'success': True,
            'deal_id': deal_id,
            'status': 'completed',
            'role': permission['role']
        })
        
    except Exception as e:
        logger.error(f"Error releasing escrow: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# -----------------------------------------------------------------------------
# CHANNEL ADMIN MANAGEMENT API
# -----------------------------------------------------------------------------

@flask_app.route('/api/channel/<int:channel_id>/admins', methods=['GET'])
def api_get_channel_admins(channel_id):
    """Get all admins for a channel"""
    try:
        admins = get_channel_admins(channel_id)
        return jsonify({'success': True, 'admins': admins})
    except Exception as e:
        logger.error(f"Error getting channel admins: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/channel/<int:channel_id>/admins', methods=['POST'])
def api_add_channel_admin(channel_id):
    """Add or update a channel admin"""
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        role = data.get('role', ChannelRole.POSTER)
        
        if not telegram_id:
            return jsonify({'success': False, 'error': 'telegram_id is required'}), 400
        
        if role not in [ChannelRole.OWNER, ChannelRole.MANAGER, ChannelRole.POSTER]:
            return jsonify({'success': False, 'error': 'Invalid role. Must be: owner, manager, or poster'}), 400
        
        # Get or create user
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
            user = cursor.fetchone()
            
            if not user:
                cursor.execute('INSERT INTO users (telegram_id) VALUES (?)', (telegram_id,))
                conn.commit()
                user_id = cursor.lastrowid
            else:
                user_id = user['id']
        
        success = set_channel_admin(channel_id, user_id, role)
        
        if success:
            return jsonify({
                'success': True,
                'channel_id': channel_id,
                'user_id': user_id,
                'role': role
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to add admin'}), 500
            
    except Exception as e:
        logger.error(f"Error adding channel admin: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/channel/<int:channel_id>/admins/<int:user_id>', methods=['DELETE'])
def api_remove_channel_admin(channel_id, user_id):
    """Remove a channel admin"""
    try:
        success = remove_channel_admin(channel_id, user_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Admin removed'})
        else:
            return jsonify({'success': False, 'error': 'Admin not found'}), 404
            
    except Exception as e:
        logger.error(f"Error removing channel admin: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/channel/<int:channel_id>/verify', methods=['POST'])
def api_verify_channel_admin(channel_id):
    """Verify admin rights via Telegram API (requires bot instance)"""
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        
        if not telegram_id:
            return jsonify({'success': False, 'error': 'telegram_id is required'}), 400
        
        if not bot_instance:
            return jsonify({'success': False, 'error': 'Bot not available for verification'}), 503
        
        # This requires async execution - return instruction for bot verification
        return jsonify({
            'success': True,
            'message': 'Verification requested. Use bot command /verify to complete.',
            'channel_id': channel_id,
            'telegram_id': telegram_id
        })
        
    except Exception as e:
        logger.error(f"Error verifying admin: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/permission/check', methods=['POST'])
def api_check_permission():
    """Check if user has permission for an action"""
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        channel_id = data.get('channel_id')
        action = data.get('action')
        
        if not all([telegram_id, channel_id, action]):
            return jsonify({'success': False, 'error': 'telegram_id, channel_id, and action are required'}), 400
        
        # Get user_id
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
            user = cursor.fetchone()
            
            if not user:
                return jsonify({'success': False, 'error': 'User not found', 'allowed': False}), 404
            
            user_id = user['id']
        
        result = check_channel_permission(user_id, channel_id, action)
        
        return jsonify({
            'success': True,
            'allowed': result['allowed'],
            'role': result['role'],
            'error': result['error']
        })
        
    except Exception as e:
        logger.error(f"Error checking permission: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/health')
def health_check():
    """Health check for Koyeb"""
    return jsonify({
        'status': 'ok',
        'service': 'tg-adescrow-bot',
        'database': DATABASE_PATH,
        'ton_escrow': TON_ESCROW_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })


# -----------------------------------------------------------------------------
# TON ESCROW API
# -----------------------------------------------------------------------------

@flask_app.route('/api/deal/<int:deal_id>/escrow/create', methods=['POST'])
def api_create_escrow_wallet(deal_id):
    """
    Create a new TON escrow wallet for a deal.
    One wallet per deal - generates unique address and stores encrypted private key.
    """
    if not TON_ESCROW_AVAILABLE:
        return jsonify({
            'success': False, 
            'error': 'TON escrow module not available. Install tonsdk, cryptography, aiohttp.'
        }), 503
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Check if deal exists
            cursor.execute('SELECT id, status, escrow_amount FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()
            
            if not deal:
                return jsonify({'success': False, 'error': 'Deal not found'}), 404
            
            # Check if wallet already exists for this deal
            cursor.execute('SELECT address FROM escrow_wallets WHERE deal_id = ?', (deal_id,))
            existing = cursor.fetchone()
            
            if existing:
                return jsonify({
                    'success': True,
                    'message': 'Escrow wallet already exists',
                    'wallet': {
                        'deal_id': deal_id,
                        'address': existing['address'],
                        'expected_amount': deal['escrow_amount']
                    }
                })
            
            # Generate new wallet
            wallet_info = ton_escrow.generate_escrow_wallet()
            
            # Store in database
            cursor.execute('''
                INSERT INTO escrow_wallets (deal_id, address, encrypted_private_key, wallet_version)
                VALUES (?, ?, ?, ?)
            ''', (
                deal_id,
                wallet_info['address'],
                wallet_info['encrypted_mnemonic'],
                wallet_info['wallet_version']
            ))
            conn.commit()
            
            wallet_id = cursor.lastrowid
            logger.info(f"Created escrow wallet {wallet_id} for deal {deal_id}: {wallet_info['address'][:20]}...")
            
            return jsonify({
                'success': True,
                'wallet': {
                    'id': wallet_id,
                    'deal_id': deal_id,
                    'address': wallet_info['address'],
                    'expected_amount': deal['escrow_amount'],
                    'network': ton_escrow.TON_NETWORK
                }
            })
            
    except Exception as e:
        logger.error(f"Error creating escrow wallet: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/escrow/status', methods=['GET'])
def api_get_escrow_status(deal_id):
    """
    Get escrow wallet status including balance and payment verification.
    """
    if not TON_ESCROW_AVAILABLE:
        return jsonify({
            'success': False, 
            'error': 'TON escrow module not available'
        }), 503
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get deal and wallet info
            cursor.execute('''
                SELECT d.id, d.status, d.escrow_amount, d.advertiser_wallet, d.channel_owner_wallet,
                       w.id as wallet_id, w.address, w.balance as cached_balance, w.last_checked
                FROM deals d
                LEFT JOIN escrow_wallets w ON d.id = w.deal_id
                WHERE d.id = ?
            ''', (deal_id,))
            row = cursor.fetchone()
            
            if not row:
                return jsonify({'success': False, 'error': 'Deal not found'}), 404
            
            if not row['address']:
                return jsonify({
                    'success': False,
                    'error': 'No escrow wallet created for this deal',
                    'hint': 'Call POST /api/deal/{id}/escrow/create first'
                }), 404
            
            # Get live balance from blockchain
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            balance_info = loop.run_until_complete(ton_escrow.get_wallet_balance(row['address']))
            loop.close()
            
            # Check if funded
            expected = row['escrow_amount'] or 0
            current_balance = balance_info.get('balance', 0)
            is_funded = current_balance >= expected * 0.99 if expected > 0 else False
            
            # Update cached balance
            cursor.execute('''
                UPDATE escrow_wallets SET balance = ?, last_checked = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (current_balance, row['wallet_id']))
            conn.commit()
            
            return jsonify({
                'success': True,
                'escrow': {
                    'deal_id': deal_id,
                    'deal_status': row['status'],
                    'address': row['address'],
                    'expected_amount': expected,
                    'current_balance': current_balance,
                    'is_funded': is_funded,
                    'network': ton_escrow.TON_NETWORK,
                    'last_checked': datetime.now().isoformat()
                }
            })
            
    except Exception as e:
        logger.error(f"Error getting escrow status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/escrow/verify', methods=['POST'])
def api_verify_escrow_deposit(deal_id):
    """
    Verify deposit and update deal status to 'funded' if amount is correct.
    """
    if not TON_ESCROW_AVAILABLE:
        return jsonify({'success': False, 'error': 'TON escrow module not available'}), 503
    
    try:
        data = request.get_json() or {}
        advertiser_wallet = data.get('advertiser_wallet')  # Optional: store sender wallet
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get deal and wallet
            cursor.execute('''
                SELECT d.id, d.status, d.escrow_amount,
                       w.id as wallet_id, w.address
                FROM deals d
                JOIN escrow_wallets w ON d.id = w.deal_id
                WHERE d.id = ?
            ''', (deal_id,))
            row = cursor.fetchone()
            
            if not row:
                return jsonify({'success': False, 'error': 'Deal or escrow wallet not found'}), 404
            
            # Check current deposit status
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            deposit_info = loop.run_until_complete(
                ton_escrow.check_for_deposit(row['address'], row['escrow_amount'])
            )
            loop.close()
            
            if deposit_info.get('funded'):
                # Update deal status to funded
                cursor.execute('''
                    UPDATE deals SET status = 'funded', advertiser_wallet = ?
                    WHERE id = ? AND status IN ('pending', 'accepted')
                ''', (advertiser_wallet or deposit_info.get('from_address'), deal_id))
                
                # Log transaction
                cursor.execute('''
                    INSERT OR IGNORE INTO escrow_transactions 
                    (wallet_id, tx_hash, tx_type, amount, from_address, to_address, status)
                    VALUES (?, ?, 'deposit', ?, ?, ?, 'confirmed')
                ''', (
                    row['wallet_id'],
                    deposit_info.get('transaction_hash'),
                    deposit_info.get('received_amount'),
                    deposit_info.get('from_address'),
                    row['address']
                ))
                
                # Update wallet balance
                cursor.execute('''
                    UPDATE escrow_wallets SET balance = ?, last_checked = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (deposit_info.get('received_amount'), row['wallet_id']))
                
                conn.commit()
                
                logger.info(f"Deal {deal_id} funded with {deposit_info.get('received_amount')} TON")
                
                return jsonify({
                    'success': True,
                    'funded': True,
                    'deal_id': deal_id,
                    'new_status': 'funded',
                    'received_amount': deposit_info.get('received_amount'),
                    'transaction_hash': deposit_info.get('transaction_hash'),
                    'from_address': deposit_info.get('from_address')
                })
            else:
                return jsonify({
                    'success': True,
                    'funded': False,
                    'deal_id': deal_id,
                    'current_status': row['status'],
                    'expected_amount': row['escrow_amount'],
                    'received_amount': deposit_info.get('received_amount', 0),
                    'message': 'Deposit not detected or amount insufficient'
                })
                
    except Exception as e:
        logger.error(f"Error verifying deposit: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/escrow/release', methods=['POST'])
def api_release_escrow_funds(deal_id):
    """
    Release escrow funds to channel owner after ad is posted/verified.
    """
    if not TON_ESCROW_AVAILABLE:
        return jsonify({'success': False, 'error': 'TON escrow module not available'}), 503
    
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        channel_owner_wallet = data.get('channel_owner_wallet')
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get deal, wallet, and channel info
            cursor.execute('''
                SELECT d.id, d.status, d.escrow_amount, d.channel_id, d.channel_owner_wallet,
                       w.id as wallet_id, w.address, w.encrypted_private_key, w.balance,
                       c.owner_ton_wallet
                FROM deals d
                JOIN escrow_wallets w ON d.id = w.deal_id
                LEFT JOIN channels c ON d.channel_id = c.id
                WHERE d.id = ?
            ''', (deal_id,))
            row = cursor.fetchone()
            
            if not row:
                return jsonify({'success': False, 'error': 'Deal or escrow wallet not found'}), 404
            
            # Check deal status allows release
            if row['status'] not in ['funded', 'posted', 'verified']:
                return jsonify({
                    'success': False,
                    'error': f"Cannot release from status '{row['status']}'. Must be funded, posted, or verified."
                }), 400
            
            # Determine destination wallet
            dest_wallet = channel_owner_wallet or row['channel_owner_wallet'] or row['owner_ton_wallet']
            
            if not dest_wallet:
                return jsonify({
                    'success': False,
                    'error': 'No destination wallet specified. Provide channel_owner_wallet in request.'
                }), 400
            
            # Get current balance
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            balance_info = loop.run_until_complete(ton_escrow.get_wallet_balance(row['address']))
            current_balance = balance_info.get('balance', 0)
            
            if current_balance <= 0.05:  # Minimum for fees
                loop.close()
                return jsonify({
                    'success': False,
                    'error': f'Insufficient balance: {current_balance} TON'
                }), 400
            
            # Release funds
            release_result = loop.run_until_complete(
                ton_escrow.release_funds(row['encrypted_private_key'], dest_wallet, current_balance)
            )
            loop.close()
            
            if release_result.get('success'):
                # Update deal status
                cursor.execute('''
                    UPDATE deals SET status = 'completed', channel_owner_wallet = ?
                    WHERE id = ?
                ''', (dest_wallet, deal_id))
                
                # Log transaction
                cursor.execute('''
                    INSERT INTO escrow_transactions 
                    (wallet_id, tx_hash, tx_type, amount, from_address, to_address, status)
                    VALUES (?, ?, 'release', ?, ?, ?, 'confirmed')
                ''', (
                    row['wallet_id'],
                    release_result.get('tx_hash'),
                    current_balance,
                    row['address'],
                    dest_wallet
                ))
                
                # Update wallet balance
                cursor.execute('''
                    UPDATE escrow_wallets SET balance = 0, last_checked = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (row['wallet_id'],))
                
                conn.commit()
                
                logger.info(f"Released {current_balance} TON from deal {deal_id} to {dest_wallet[:20]}...")
                
                return jsonify({
                    'success': True,
                    'deal_id': deal_id,
                    'new_status': 'completed',
                    'released_amount': current_balance,
                    'destination': dest_wallet,
                    'tx_hash': release_result.get('tx_hash')
                })
            else:
                return jsonify({
                    'success': False,
                    'error': release_result.get('error', 'Release transaction failed')
                }), 500
                
    except Exception as e:
        logger.error(f"Error releasing escrow: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/escrow/refund', methods=['POST'])
def api_refund_escrow(deal_id):
    """
    Refund escrow funds to advertiser.
    """
    if not TON_ESCROW_AVAILABLE:
        return jsonify({'success': False, 'error': 'TON escrow module not available'}), 503
    
    try:
        data = request.get_json() or {}
        advertiser_wallet = data.get('advertiser_wallet')
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get deal and wallet info
            cursor.execute('''
                SELECT d.id, d.status, d.escrow_amount, d.advertiser_wallet,
                       w.id as wallet_id, w.address, w.encrypted_private_key, w.balance
                FROM deals d
                JOIN escrow_wallets w ON d.id = w.deal_id
                WHERE d.id = ?
            ''', (deal_id,))
            row = cursor.fetchone()
            
            if not row:
                return jsonify({'success': False, 'error': 'Deal or escrow wallet not found'}), 404
            
            # Check status allows refund
            if row['status'] not in ['funded', 'posted', 'verified']:
                return jsonify({
                    'success': False,
                    'error': f"Cannot refund from status '{row['status']}'. Must be funded, posted, or verified."
                }), 400
            
            # Determine refund destination
            dest_wallet = advertiser_wallet or row['advertiser_wallet']
            
            if not dest_wallet:
                return jsonify({
                    'success': False,
                    'error': 'No advertiser wallet known. Provide advertiser_wallet in request.'
                }), 400
            
            # Get current balance
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            balance_info = loop.run_until_complete(ton_escrow.get_wallet_balance(row['address']))
            current_balance = balance_info.get('balance', 0)
            
            if current_balance <= 0.05:
                loop.close()
                return jsonify({
                    'success': False,
                    'error': f'Insufficient balance for refund: {current_balance} TON'
                }), 400
            
            # Refund
            refund_result = loop.run_until_complete(
                ton_escrow.refund_funds(row['encrypted_private_key'], dest_wallet, current_balance)
            )
            loop.close()
            
            if refund_result.get('success'):
                # Update deal status
                cursor.execute('UPDATE deals SET status = ? WHERE id = ?', ('refunded', deal_id))
                
                # Log transaction
                cursor.execute('''
                    INSERT INTO escrow_transactions 
                    (wallet_id, tx_hash, tx_type, amount, from_address, to_address, status)
                    VALUES (?, ?, 'refund', ?, ?, ?, 'confirmed')
                ''', (
                    row['wallet_id'],
                    refund_result.get('tx_hash'),
                    current_balance,
                    row['address'],
                    dest_wallet
                ))
                
                # Update wallet balance
                cursor.execute('''
                    UPDATE escrow_wallets SET balance = 0, last_checked = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (row['wallet_id'],))
                
                conn.commit()
                
                logger.info(f"Refunded {current_balance} TON from deal {deal_id} to {dest_wallet[:20]}...")
                
                return jsonify({
                    'success': True,
                    'deal_id': deal_id,
                    'new_status': 'refunded',
                    'refunded_amount': current_balance,
                    'destination': dest_wallet,
                    'tx_hash': refund_result.get('tx_hash')
                })
            else:
                return jsonify({
                    'success': False,
                    'error': refund_result.get('error', 'Refund transaction failed')
                }), 500
                
    except Exception as e:
        logger.error(f"Error refunding escrow: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/escrow/transactions', methods=['GET'])
def api_get_escrow_transactions(deal_id):
    """Get all transactions for a deal's escrow wallet"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT et.id, et.tx_hash, et.tx_type, et.amount, 
                       et.from_address, et.to_address, et.status, et.created_at
                FROM escrow_transactions et
                JOIN escrow_wallets ew ON et.wallet_id = ew.id
                WHERE ew.deal_id = ?
                ORDER BY et.created_at DESC
            ''', (deal_id,))
            rows = cursor.fetchall()
            
            transactions = [dict(row) for row in rows]
            
            return jsonify({
                'success': True,
                'deal_id': deal_id,
                'transactions': transactions
            })
            
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# -----------------------------------------------------------------------------
# AUTO-POSTER API
# -----------------------------------------------------------------------------

@flask_app.route('/api/deal/<int:deal_id>/post/schedule', methods=['POST'])
def api_schedule_post(deal_id):
    """
    Schedule an ad post for a future time.
    Body: {scheduled_time: ISO datetime, ad_text: string, hold_hours: int (optional)}
    """
    if not AUTO_POSTER_AVAILABLE:
        return jsonify({'success': False, 'error': 'Auto-poster module not available'}), 503
    
    try:
        data = request.get_json() or {}
        scheduled_time_str = data.get('scheduled_time')
        ad_text = data.get('ad_text')
        hold_hours = data.get('hold_hours', 24)
        
        if not scheduled_time_str:
            return jsonify({'success': False, 'error': 'scheduled_time is required'}), 400
        if not ad_text:
            return jsonify({'success': False, 'error': 'ad_text is required'}), 400
        
        # Parse scheduled time
        try:
            scheduled_time = datetime.fromisoformat(scheduled_time_str.replace('Z', '+00:00'))
        except Exception:
            return jsonify({'success': False, 'error': 'Invalid scheduled_time format'}), 400
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get deal and channel info
            cursor.execute('''
                SELECT d.id, d.status, d.channel_id, c.telegram_channel_id, c.bot_can_post
                FROM deals d
                JOIN channels c ON d.channel_id = c.id
                WHERE d.id = ?
            ''', (deal_id,))
            deal = cursor.fetchone()
            
            if not deal:
                return jsonify({'success': False, 'error': 'Deal not found'}), 404
            
            if deal['status'] not in ['funded', 'accepted']:
                return jsonify({
                    'success': False,
                    'error': f"Cannot schedule from status '{deal['status']}'. Deal must be funded."
                }), 400
            
            if not deal['bot_can_post']:
                return jsonify({
                    'success': False,
                    'error': 'Bot cannot post to this channel. Verify bot is admin with posting rights.'
                }), 400
        
        # Schedule the post
        result = auto_poster.schedule_post(
            deal_id=deal_id,
            channel_id=deal['channel_id'],
            ad_text=ad_text,
            scheduled_time=scheduled_time,
            hold_hours=hold_hours
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'post_id': result['post_id'],
                'scheduled_time': scheduled_time_str,
                'hold_hours': hold_hours,
                'message': f'Post scheduled for {scheduled_time_str}'
            })
        else:
            return jsonify({'success': False, 'error': result['error']}), 400
            
    except Exception as e:
        logger.error(f"Error scheduling post: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/post/now', methods=['POST'])
def api_post_now(deal_id):
    """Post ad immediately"""
    if not AUTO_POSTER_AVAILABLE:
        return jsonify({'success': False, 'error': 'Auto-poster module not available'}), 503
    
    try:
        data = request.get_json() or {}
        ad_text = data.get('ad_text')
        hold_hours = data.get('hold_hours', 24)
        
        if not ad_text:
            return jsonify({'success': False, 'error': 'ad_text is required'}), 400
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.id, d.status, d.channel_id, c.telegram_channel_id, c.bot_can_post
                FROM deals d
                JOIN channels c ON d.channel_id = c.id
                WHERE d.id = ?
            ''', (deal_id,))
            deal = cursor.fetchone()
            
            if not deal:
                return jsonify({'success': False, 'error': 'Deal not found'}), 404
            
            if not deal['telegram_channel_id']:
                return jsonify({'success': False, 'error': 'Channel not verified'}), 400
            
            if not deal['bot_can_post']:
                return jsonify({'success': False, 'error': 'Bot cannot post to channel'}), 400
        
        # Post immediately using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            auto_poster.post_to_channel(bot_instance.app.bot, deal['telegram_channel_id'], ad_text)
        )
        loop.close()
        
        if result['success']:
            now = datetime.now()
            release_at = now + timedelta(hours=hold_hours)
            
            # Create scheduled_posts entry
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO scheduled_posts 
                    (deal_id, channel_id, ad_text, scheduled_time, posted_at, message_id, 
                     hold_hours, release_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'posted')
                ''', (deal_id, deal['channel_id'], ad_text, now.isoformat(), 
                      now.isoformat(), result['message_id'], hold_hours, release_at.isoformat()))
                
                cursor.execute('''
                    UPDATE deals SET status = 'posted', message_id = ?, posted_at = ?
                    WHERE id = ?
                ''', (result['message_id'], now.isoformat(), deal_id))
                
                conn.commit()
            
            return jsonify({
                'success': True,
                'message_id': result['message_id'],
                'posted_at': now.isoformat(),
                'release_at': release_at.isoformat(),
                'hold_hours': hold_hours
            })
        else:
            return jsonify({'success': False, 'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"Error posting now: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/post/verify', methods=['GET'])
def api_verify_post(deal_id):
    """Check if post still exists in channel"""
    if not AUTO_POSTER_AVAILABLE:
        return jsonify({'success': False, 'error': 'Auto-poster module not available'}), 503
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT sp.message_id, sp.status, sp.posted_at, sp.release_at,
                       c.telegram_channel_id
                FROM scheduled_posts sp
                JOIN channels c ON sp.channel_id = c.id
                WHERE sp.deal_id = ?
            ''', (deal_id,))
            post = cursor.fetchone()
            
            if not post:
                return jsonify({'success': False, 'error': 'No post found for this deal'}), 404
            
            if not post['message_id']:
                return jsonify({
                    'success': True,
                    'status': post['status'],
                    'exists': None,
                    'message': 'Post not yet sent'
                })
        
        # Verify message exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            auto_poster.verify_message_exists(
                bot_instance.app.bot, 
                post['telegram_channel_id'], 
                post['message_id']
            )
        )
        loop.close()
        
        return jsonify({
            'success': True,
            'deal_id': deal_id,
            'message_id': post['message_id'],
            'exists': result['exists'],
            'status': post['status'],
            'posted_at': post['posted_at'],
            'release_at': post['release_at'],
            'error': result.get('error')
        })
        
    except Exception as e:
        logger.error(f"Error verifying post: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@flask_app.route('/api/deal/<int:deal_id>/post/cancel', methods=['POST'])
def api_cancel_scheduled_post(deal_id):
    """Cancel a scheduled (not yet posted) post"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, status FROM scheduled_posts WHERE deal_id = ?',
                (deal_id,)
            )
            post = cursor.fetchone()
            
            if not post:
                return jsonify({'success': False, 'error': 'No post found'}), 404
            
            if post['status'] != 'scheduled':
                return jsonify({
                    'success': False,
                    'error': f"Cannot cancel post with status '{post['status']}'"
                }), 400
            
            cursor.execute(
                'DELETE FROM scheduled_posts WHERE id = ?',
                (post['id'],)
            )
            cursor.execute(
                "UPDATE deals SET status = 'funded' WHERE id = ?",
                (deal_id,)
            )
            conn.commit()
            
        return jsonify({
            'success': True,
            'message': 'Scheduled post cancelled'
        })
        
    except Exception as e:
        logger.error(f"Error cancelling post: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# MAIN
# =============================================================================

def run_flask():
    """Run Flask server"""
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Flask server on port {port}")
    flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


def main():
    """Main entry point"""
    global bot_instance
    
    # Initialize database
    init_database()
    
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("BOT_TOKEN environment variable is required")
    
    bot_instance = AdEscrowBot(token)
    
    # Start Flask in background
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Flask server started")
    
    # Start auto-poster scheduler
    if AUTO_POSTER_AVAILABLE:
        auto_poster.start_scheduler(bot_instance.app)
        logger.info("Auto-poster scheduler started")
    
    # Run bot (blocking)
    bot_instance.run()


if __name__ == "__main__":
    main()


