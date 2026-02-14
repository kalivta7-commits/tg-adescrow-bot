import json
import logging
import os
import sqlite3
import threading
import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
from functools import wraps

from dotenv import load_dotenv
from supabase import create_client
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import TON escrow module (optional - graceful degradation if dependencies missing)
try:
    import ton_escrow
    from ton_escrow import sync_send_from_platform_wallet
    TON_ESCROW_AVAILABLE = True
except ImportError as e:
    TON_ESCROW_AVAILABLE = False
    sync_send_from_platform_wallet = None
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
TOKEN = os.getenv("BOT_TOKEN")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        logger.warning("Supabase env vars missing; upload API disabled")
except Exception as e:
    logger.warning(f"Supabase client init failed: {e}")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MIN_ESCROW_BALANCE = 0.05
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

# -----------------------------------------------------------------------------
# Async helper
# -----------------------------------------------------------------------------
async def run_async(coro):
    """Safely run a coroutine from a synchronous context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create new one
        return asyncio.run(coro)
    else:
        # Use existing loop
        return await coro


def run_coroutine_safely(coro):
    """Run coroutine from sync code without event-loop conflicts."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        result_box = {'value': None, 'error': None}

        def _runner():
            try:
                result_box['value'] = asyncio.run(coro)
            except Exception as inner_err:
                result_box['error'] = inner_err

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()
        worker.join()

        if result_box['error']:
            raise result_box['error']

        return result_box['value']


def dispatch_background_async(coro, task_name: str = "background-task"):
    """Execute async call in background thread and never crash request thread."""
    def _runner():
        try:
            run_coroutine_safely(coro)
        except Exception as async_err:
            logger.error("Async task %s failed: %s", task_name, async_err, exc_info=True)

    threading.Thread(target=_runner, daemon=True).start()

# -----------------------------------------------------------------------------
# Rate limiting stub
# -----------------------------------------------------------------------------
def rate_limit(limit=5, per=60):
    """Simple rate limiting decorator (stub)."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # TODO: implement actual rate limiting (e.g., using Redis)
            return f(*args, **kwargs)
        return wrapped
    return decorator

# =============================================================================
# DATABASE SETUP
# =============================================================================

DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')


def init_database():
    """Initialize SQLite database with required tables and indexes"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Enable WAL mode for better concurrency
    cursor.execute('PRAGMA journal_mode=WAL')
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
            public_link TEXT,
            name TEXT,
            category TEXT DEFAULT 'general',
            price REAL DEFAULT 0,
            subscribers INTEGER DEFAULT 0,
            avg_views INTEGER DEFAULT 0,
            verified INTEGER DEFAULT 0,
            bot_is_admin INTEGER DEFAULT 0,
            bot_can_post INTEGER DEFAULT 0,
            owner_ton_wallet TEXT,
            owner_wallet TEXT,
            total_deals INTEGER DEFAULT 0,
            completed_deals INTEGER DEFAULT 0,
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
            paid_at TIMESTAMP,
            ad_posted_at TIMESTAMP,
            release_at TIMESTAMP,
            refunded_at TIMESTAMP,
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
        'ALTER TABLE channels ADD COLUMN public_link TEXT',
        'ALTER TABLE channels ADD COLUMN owner_ton_wallet TEXT',
        'ALTER TABLE channels ADD COLUMN owner_wallet TEXT',
        'ALTER TABLE channels ADD COLUMN total_deals INTEGER DEFAULT 0',
        'ALTER TABLE channels ADD COLUMN completed_deals INTEGER DEFAULT 0',
        'ALTER TABLE deals ADD COLUMN advertiser_wallet TEXT',
        'ALTER TABLE deals ADD COLUMN channel_owner_wallet TEXT',
        'ALTER TABLE deals ADD COLUMN message_id INTEGER',
        'ALTER TABLE deals ADD COLUMN posted_message_id INTEGER',
        'ALTER TABLE deals ADD COLUMN posted_chat_id INTEGER',
        'ALTER TABLE deals ADD COLUMN posted_at TIMESTAMP',
        'ALTER TABLE deals ADD COLUMN hold_hours INTEGER DEFAULT 24',
        'ALTER TABLE users ADD COLUMN ton_wallet TEXT',
        'ALTER TABLE deals ADD COLUMN media_type TEXT',
        'ALTER TABLE deals ADD COLUMN media_url TEXT',
        'ALTER TABLE deals ADD COLUMN paid_at TIMESTAMP',
        'ALTER TABLE deals ADD COLUMN ad_posted_at TIMESTAMP',
        'ALTER TABLE deals ADD COLUMN release_at TIMESTAMP',
        'ALTER TABLE deals ADD COLUMN refunded_at TIMESTAMP',
    ]

    for migration in migrations:
        try:
            cursor.execute(migration)
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_deals_status ON deals(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_deals_channel ON deals(channel_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_campaigns_advertiser ON campaigns(advertiser_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_channels_owner ON channels(owner_id)')

    conn.commit()
    conn.close()
    logger.info(f'Database initialized at {DATABASE_PATH}')


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    # Ensure WAL and foreign keys are enabled for every connection
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA journal_mode = WAL')
    try:
        yield conn
    finally:
        conn.close()


def get_user_id(telegram_id: int) -> Optional[int]:
    """Get database user_id from telegram_id, creating user if not exists."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
        row = cursor.fetchone()
        if row:
            return row['id']
        # Auto-register
        cursor.execute('INSERT INTO users (telegram_id) VALUES (?)', (telegram_id,))
        conn.commit()
        return cursor.lastrowid


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
        user_id = get_user_id(telegram_user_id)

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
        if not channel_username or not isinstance(channel_username, str):
            result['error'] = 'Invalid username. Provide a valid public @username.'
            return result

        # Ensure @ prefix
        if not channel_username.startswith('@'):
            channel_username = '@' + channel_username

        if ' ' in channel_username or len(channel_username) < 2:
            result['error'] = 'Invalid username format. Use a public channel @username.'
            return result

        # Get channel info
        try:
            chat = await bot.get_chat(channel_username)
        except Exception as e:
            error_str = str(e).lower()
            if 'chat not found' in error_str:
                result['error'] = 'Channel not found. Check the username.'
            elif 'username is invalid' in error_str or 'invalid' in error_str:
                result['error'] = 'Invalid username. Check channel handle and try again.'
            elif 'private' in error_str:
                result['error'] = 'Private channel is not supported. Use a public channel username.'
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

        # Check if bot is admin and can post
        try:
            bot_member = await bot.get_chat_member(chat.id, bot.id)
            status = getattr(bot_member, 'status', None)

            if status not in ('administrator', 'creator'):
                result['error'] = 'Bot is not an admin of this channel. Add bot as admin with "Post Messages" permission.'
                return result

            result['bot_is_admin'] = True

            # Creators can always post; administrators require explicit post permission.
            if status == 'creator':
                result['bot_can_post'] = True
            else:
                result['bot_can_post'] = bool(getattr(bot_member, 'can_post_messages', False))

            if not result['bot_can_post']:
                result['error'] = 'Bot is admin but cannot post messages. Enable "Post Messages" permission.'
                return result

            result['verified'] = True
            result['success'] = True
            return result

        except Exception as e:
            error_str = str(e).lower()
            if 'bot was kicked' in error_str:
                result['error'] = 'Bot was kicked from the channel. Add bot back as admin.'
            elif 'not enough rights' in error_str or 'have no rights' in error_str:
                result['error'] = 'Bot lacks admin rights in the channel.'
            else:
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
                                       category: str = 'general', price: float = 0,
                                       owner_wallet: str = None) -> dict:
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

    clean_username = (verification.get('username') or channel_username).strip().lstrip('@')
    public_link = f"https://t.me/{clean_username}" if clean_username else None

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
                        username = ?,
                        public_link = ?,
                        name = ?,
                        subscribers = ?,
                        verified = 1,
                        bot_is_admin = 1,
                        bot_can_post = 1,
                        owner_ton_wallet = COALESCE(?, owner_ton_wallet),
                        owner_wallet = COALESCE(?, owner_wallet),
                        verified_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    verification['telegram_channel_id'],
                    channel_username,
                    public_link,
                    verification['title'],
                    verification['subscribers'],
                    owner_wallet,
                    owner_wallet,
                    existing['id']
                ))
                conn.commit()
                channel_id = existing['id']
                logger.info(f"Updated existing channel {channel_id} - {channel_username}")
            else:
                # Create new channel
                cursor.execute('''
                    INSERT INTO channels (
                        owner_id, telegram_channel_id, username, public_link, name, category, price,
                        subscribers, verified, bot_is_admin, bot_can_post, owner_ton_wallet, owner_wallet, verified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 1, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    owner_id,
                    verification['telegram_channel_id'],
                    channel_username,
                    public_link,
                    verification['title'],
                    category,
                    price,
                    verification['subscribers'],
                    owner_wallet,
                    owner_wallet
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
                'owner_wallet': owner_wallet,
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
    DELETED = "deleted"


# =============================================================================
# DEAL STATE MACHINE
# =============================================================================

class DealStateMachine:
    """
    Strict state machine for deal transitions.
    Ensures atomic, valid state changes with logging.
    """

    TERMINAL_STATES = ['completed', 'cancelled', 'refunded', 'deleted']

    # Valid state transitions: current_state -> [allowed_next_states]
    TRANSITIONS = {
        'pending': ['accepted', 'cancelled'],
        'accepted': ['funded', 'cancelled'],
        'funded': ['scheduled', 'posted', 'refunded'],
        'scheduled': ['posted', 'cancelled', 'refunded'],
        'posted': ['verified', 'refunded'],
        'verified': ['completed', 'refunded'],
        'completed': [],
        'refunded': [],
        'cancelled': [],
        'deleted': [],
    }

    STATE_LABELS = {
        'pending': 'Pending Approval',
        'accepted': 'Accepted',
        'funded': 'Escrow Funded',
        'scheduled': 'Post Scheduled',
        'posted': 'Ad Posted',
        'verified': 'Verified',
        'completed': 'Completed',
        'refunded': 'Refunded',
        'cancelled': 'Cancelled',
        'deleted': 'Deleted'
    }

    STATE_STEPS = {
        'pending': 1,
        'accepted': 2,
        'funded': 3,
        'posted': 4,
        'verified': 5,
        'completed': 6,
        'refunded': 0,
        'cancelled': 0,
        'deleted': 0
    }

    @classmethod
    def can_transition(cls, current_state: str, new_state: str) -> bool:
        allowed = cls.TRANSITIONS.get(current_state, [])
        return new_state in allowed

    @classmethod
    def get_allowed_transitions(cls, current_state: str) -> List[str]:
        return cls.TRANSITIONS.get(current_state, [])

    @classmethod
    def is_terminal(cls, state: str) -> bool:
        return state in cls.TERMINAL_STATES

    @classmethod
    def get_step(cls, state: str) -> int:
        return cls.STATE_STEPS.get(state, 1)

    @classmethod
    def get_label(cls, state: str) -> str:
        return cls.STATE_LABELS.get(state, state.title())


DEAL_ACTION_TO_STATE = {
    'fund': 'funded',
    'cancel': 'cancelled',
    'verify': 'verified',
    'dispute': 'refunded',
    'accept': 'accepted',
    'reject': 'cancelled',
    'mark_posted': 'posted',
    'delete': 'deleted'
}

DEAL_STATE_TO_ACTION = {
    'funded': 'fund',
    'cancelled': 'cancel',
    'verified': 'verify',
    'refunded': 'dispute',
    'accepted': 'accept',
    'posted': 'mark_posted',
    'deleted': 'delete'
}


def get_role_allowed_actions(state: str, role: str) -> List[str]:
    all_transitions = DealStateMachine.get_allowed_transitions(state)
    all_actions = [DEAL_STATE_TO_ACTION[t] for t in all_transitions if t in DEAL_STATE_TO_ACTION]

    if role == 'advertiser':
        return [t for t in all_actions if t in ['fund', 'cancel', 'verify', 'dispute']]
    if role == 'owner':
        return [t for t in all_actions if t in ['accept', 'reject', 'mark_posted', 'delete']]
    return []


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
        'new_state': None,
        'conflict': False
    }

    try:
        with get_db() as conn:
            # Begin transaction immediately for atomicity
            conn.execute('BEGIN IMMEDIATE')
            cursor = conn.cursor()

            # Get current deal state (locked by transaction)
            cursor.execute('SELECT * FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()

            if not deal:
                conn.rollback()
                result['error'] = 'Deal not found'
                return result

            current_state = deal['status']
            result['old_state'] = current_state

            if not DealStateMachine.can_transition(current_state, new_state):
                allowed = DealStateMachine.get_allowed_transitions(current_state)
                conn.rollback()
                result['error'] = f"Invalid transition: {current_state} → {new_state}. Allowed: {allowed}"
                return result

            # Optimistic update using current_state check
            if new_state == 'paid':
                cursor.execute('''
                    UPDATE deals
                    SET status = ?, paid_at = ?
                    WHERE id = ? AND status = ?
                ''', (new_state, datetime.utcnow().isoformat(), deal_id, current_state))
            else:
                cursor.execute('''
                    UPDATE deals SET status = ? WHERE id = ? AND status = ?
                ''', (new_state, deal_id, current_state))

            if cursor.rowcount == 0:
                conn.rollback()
                result['error'] = 'State changed by another process (concurrent modification)'
                result['conflict'] = True
                return result

            if new_state == 'completed' and current_state != 'completed':
                cursor.execute('''
                    UPDATE channels
                    SET completed_deals = COALESCE(completed_deals, 0) + 1
                    WHERE id = ?
                ''', (deal['channel_id'],))

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
        logger.error(f"Deal transition error: {e}")
        result['error'] = f'Database error: {str(e)}'

    return result


async def send_deal_notification(bot, deal_id: int, event_type: str, extra_data: dict = None):
    """
    Send notification for a deal event.
    """
    if not NOTIFICATIONS_AVAILABLE:
        logger.debug("Notifications module not available, skipping notification")
        return

    try:
        deal_data = notifications.get_deal_data_for_notification(deal_id)
        if not deal_data:
            logger.warning(f"Could not get deal data for notification: deal_id={deal_id}")
            return

        if extra_data:
            deal_data.update(extra_data)

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
    CREATE_CAMPAIGN = "create_campaign"
    ADD_CHANNEL = "add_channel"
    SELECT_CHANNELS = "select_channels"
    VIEW_MARKETPLACE = "view_marketplace"
    MANAGE_CAMPAIGNS = "manage_campaigns"
    MANAGE_CHANNELS = "manage_channels"


@dataclass
class Campaign:
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
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("menu", self.menu_command))
        self.application.add_handler(CallbackQueryHandler(self.handle_button_callback))
        self.application.add_handler(
            MessageHandler(filters.StatusUpdate.WEB_APP_DATA, self.handle_webapp_data)
        )
        self.application.add_error_handler(self.error_handler)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        welcome_text = (
            "🤖 *Welcome to TG AdEscrow Bot!*\n\n"
            "Your trusted marketplace for Telegram advertising with escrow protection.\n\n"
            "🔐 *Secure Transactions*\n"
            "💰 *Transparent Pricing*\n"
            "📊 *Real-time Analytics*\n\n"
            "Click the button below to open the Mini App and get started!"
        )

        webapp_url = os.getenv("WEBAPP_URL", "")
        if not webapp_url:
            webapp_url = os.getenv("KOYEB_PUBLIC_DOMAIN", "")
            if webapp_url and not webapp_url.startswith("http"):
                webapp_url = f"https://{webapp_url}"

        keyboard = []
        if webapp_url:
            keyboard.append([InlineKeyboardButton(
                text="🚀 Open Ad Marketplace",
                web_app=WebAppInfo(url=webapp_url)
            )])
        keyboard.append([InlineKeyboardButton("📋 Help Guide", callback_data="help")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        help_text = self.get_help_guide_text()
        await update.message.reply_text(help_text)

    def get_help_guide_text(self) -> str:
        return (
            "📖 TG AdEscrow – Complete User Guide\n\n"
            "Welcome to TG AdEscrow — a secure marketplace for Telegram advertising using TON escrow protection.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🚀 FOR ADVERTISERS (BUYERS)\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "🔹 Step 1 – Open Marketplace\n"
            "Tap the 🚀 Open Marketplace button to browse available Telegram channels.\n\n"
            "🔹 Step 2 – Browse & Filter\n"
            "You can filter channels by:\n"
            "• Category\n"
            "• Minimum subscribers\n"
            "• Price\n\n"
            "🔹 Step 3 – View Channel Details\n"
            "Each listing shows:\n"
            "• Subscriber count\n"
            "• Price per post\n"
            "• Category\n"
            "• Channel description\n\n"
            "🔹 Step 4 – Create a Deal\n"
            "Select a channel and create a deal.\n"
            "Provide:\n"
            "• Ad text\n"
            "• Post duration\n"
            "• Special instructions (optional)\n\n"
            "🔹 Step 5 – Pay with TON (Escrow)\n"
            "Once confirmed, you pay in TON.\n"
            "Funds are locked securely in escrow.\n"
            "The seller CANNOT access funds yet.\n\n"
            "🔹 Step 6 – Seller Posts Your Ad\n"
            "The channel owner must publish your ad exactly as agreed.\n"
            "The post must remain live for the agreed duration.\n\n"
            "🔹 Step 7 – Verification\n"
            "The bot verifies the post.\n"
            "If valid → deal moves to completion.\n\n"
            "🔹 Step 8 – Payment Release\n"
            "After successful verification, escrow releases TON to the seller.\n"
            "Deal marked as ✅ Completed.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📢 FOR CHANNEL OWNERS (SELLERS)\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "🔹 Step 1 – Register Your Channel\n"
            "Open Mini App and click Add Channel.\n"
            "You must be an admin of the channel.\n\n"
            "🔹 Step 2 – Add Bot as Admin\n"
            "To enable verification, add the bot as admin in your channel.\n\n"
            "🔹 Step 3 – Set Pricing & Details\n"
            "Define:\n"
            "• Category\n"
            "• Price per post\n"
            "• Minimum duration\n"
            "• Channel description\n\n"
            "🔹 Step 4 – Receive Deal Requests\n"
            "When an advertiser creates a deal, you’ll be notified.\n"
            "You can Accept or Reject.\n\n"
            "🔹 Step 5 – Post the Advertisement\n"
            "After escrow is funded, post the ad exactly as provided.\n"
            "Keep it live for the agreed duration.\n\n"
            "🔹 Step 6 – Receive Payment\n"
            "Once verified, TON is automatically released to your wallet.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🔐 HOW ESCROW WORKS\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "1️⃣ Buyer pays TON\n"
            "2️⃣ Funds locked in escrow\n"
            "3️⃣ Seller posts advertisement\n"
            "4️⃣ Bot verifies post\n"
            "5️⃣ Funds released automatically\n\n"
            "No middleman. No manual trust required.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "⚖️ DISPUTE SYSTEM\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "If:\n"
            "• Ad not posted\n"
            "• Ad removed early\n"
            "• Wrong content posted\n\n"
            "Buyer may raise a dispute.\n"
            "Funds remain locked until resolution.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📊 DEAL STATUS MEANINGS\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "🟡 Pending – Waiting for seller action\n"
            "🔒 Escrow Funded – Payment locked securely\n"
            "📢 Live – Advertisement currently active\n"
            "✅ Completed – Payment released\n"
            "❌ Cancelled – Deal cancelled\n"
            "⚠ Disputed – Under review\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🌟 WHY USE TG AdEscrow?\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "• Secure TON escrow protection\n"
            "• Verified channel listings\n"
            "• Transparent pricing\n"
            "• Automated deal tracking\n"
            "• Designed specifically for Telegram\n\n"
            "Safe • Transparent • Automated"
        )

    def get_support_text(self) -> str:
        return (
            "Telegram: @ejag78\n"
            "X (Twitter): @EJDEVX\n"
            "Email: ejfxprotrade@gmail.com"
        )

    async def handle_button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()

        if query.data == "help":
            await query.message.reply_text(self.get_help_guide_text())
        elif query.data == "support":
            await query.message.reply_text(self.get_support_text())

    async def handle_webapp_data(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        campaign_id = data.get('campaign_id', '')
        selected_channels = data.get('channels', [])

        if campaign_id in self.campaigns:
            campaign = self.campaigns[campaign_id]
            campaign.selected_channels = selected_channels
            campaign.escrow_status = EscrowStatus.FUNDED

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
        logger.info("Starting TG AdEscrow Bot...")
        self.application.run_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )


# =============================================================================
# FLASK APP AND API ENDPOINTS (SQLite-backed)
# =============================================================================

flask_app = Flask(__name__, static_folder='miniapp')

@flask_app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return response


bot_instance = AdEscrowBot(TOKEN) if TOKEN else None


def json_response(success: bool, data=None, error=None, status=200):
    """Standardized JSON response."""
    return jsonify({
        'success': success,
        'data': data,
        'error': error
    }), status


@flask_app.route('/')
def serve_miniapp():
    return send_from_directory(flask_app.static_folder, 'index.html')


@flask_app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(flask_app.static_folder, filename)


@flask_app.route('/api/upload', methods=['POST'])
@rate_limit()
def api_upload_media():
    try:
        if supabase is None:
            return json_response(False, error="Supabase is not configured", status=503)

        if 'file' not in request.files:
            return json_response(False, error="No file uploaded", status=400)

        file = request.files['file']
        if not file:
            return json_response(False, error="Invalid file", status=400)

        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > MAX_UPLOAD_SIZE:
            return json_response(False, error="File too large. Maximum size is 50MB.", status=400)

        filename = secure_filename((file.filename or '').lower())
        if not filename:
            return json_response(False, error="Invalid file", status=400)

        if filename.endswith(('.png','.jpg','.jpeg','.webp','.gif')):
            media_type = "photo"
        elif filename.endswith(('.mp4','.mov','.avi','.mkv')):
            media_type = "video"
        else:
            return json_response(False, error="Unsupported file type", status=400)

        import uuid
        unique_name = f"{uuid.uuid4()}_{filename}"
        file_bytes = file.read()

        response = supabase.storage.from_("ads").upload(unique_name, file_bytes, {"content-type": file.content_type})
        if isinstance(response, dict) and response.get("error"):
            return json_response(False, error=str(response["error"]), status=500)

        public = supabase.storage.from_("ads").get_public_url(unique_name)
        public_url = public.get("publicUrl") or public.get("public_url") or public.get("data") if isinstance(public, dict) else public

        return json_response(True, data={"media_type": media_type, "media_url": public_url})

    except Exception as e:
        return json_response(False, error=str(e), status=500)


# -----------------------------------------------------------------------------
# AUTH API
# -----------------------------------------------------------------------------

@flask_app.route('/api/auth', methods=['POST'])
@rate_limit()
def api_auth():
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')

        if not telegram_id:
            return json_response(False, error='telegram_id is required', status=400)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE telegram_id = ?', (telegram_id,))
            row = cursor.fetchone()

            if row:
                user = dict(row)
            else:
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

            return json_response(True, data={'user': user})

    except Exception as e:
        logger.error(f"Error in auth: {e}")
        return json_response(False, error=str(e), status=500)


# -----------------------------------------------------------------------------
# CHANNELS API
# -----------------------------------------------------------------------------

@flask_app.route('/api/channels', methods=['GET'])
@rate_limit()
def api_get_channels():
    try:
        owner_only = request.args.get('owner_only') == '1'
        telegram_id = request.args.get('telegram_id') or request.args.get('user_id')

        owner_id = None
        if owner_only:
            if not telegram_id:
                return jsonify({'success': False, 'error': 'telegram_id is required when owner_only=1'}), 400
            try:
                owner_id = get_user_id(int(telegram_id))
            except (TypeError, ValueError):
                return jsonify({'success': False, 'error': 'telegram_id must be an integer'}), 400

        with get_db() as conn:
            cursor = conn.cursor()

            if owner_only:
                cursor.execute('''
                    SELECT id, owner_id, username, public_link, name, category, price, subscribers, avg_views,
                           total_deals, completed_deals, created_at
                    FROM channels
                    WHERE owner_id = ?
                    ORDER BY subscribers DESC
                ''', (owner_id,))
            else:
                cursor.execute('''
                    SELECT id, owner_id, username, public_link, name, category, price, subscribers, avg_views,
                           total_deals, completed_deals, created_at
                    FROM channels
                    WHERE verified = 1
                    ORDER BY subscribers DESC
                ''')
            rows = cursor.fetchall()

            channels = []
            for row in rows:
                total = row['total_deals'] or 0
                completed = row['completed_deals'] or 0
                success_rate = 0
                if total > 0:
                    success_rate = round((completed / total) * 100, 2)

                channels.append({
                    'id': row['id'],
                    'owner_id': row['owner_id'],
                    'handle': row['username'],
                    'username': row['username'],
                    'public_link': row['public_link'] or (
                        f"https://t.me/{row['username'].lstrip('@')}" if row['username'] else None
                    ),
                    'name': row['name'] or row['username'],
                    'category': row['category'],
                    'price': row['price'],
                    'subscribers': row['subscribers'],
                    'views': row['avg_views'],
                    'total_deals': total,
                    'completed_deals': completed,
                    'success_rate': success_rate,
                    'created_at': row['created_at']
                })

            return jsonify({'success': True, 'data': channels}), 200

    except Exception as e:
        logger.error(f"Error getting channels: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/channels', methods=['POST'])
@flask_app.route('/api/register-channel', methods=['POST'])
@rate_limit()
def api_create_channel():
    try:
        data = request.get_json() or {}

        telegram_id = data.get('telegram_id') or data.get('user_id')
        username = data.get('username') or data.get('channel_handle')

        if not telegram_id or not username:
            return json_response(False, error='telegram_id and username are required', status=400)

        if bot_instance is None or bot_instance.application is None:
            return json_response(False, error='Bot instance is not initialized', status=503)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE telegram_id = ?", (telegram_id,))
            user = cursor.fetchone()

            if not user:
                return json_response(False, error='User not found', status=404)

            owner_id = user['id']

        category = data.get('category', 'general')
        owner_wallet = (data.get('owner_wallet') or '').strip()
        if not owner_wallet or not owner_wallet.startswith('EQ'):
            return jsonify({
                "success": False,
                "error": result.get("error", "Unknown registration error")
            }), 400

        allowed_categories = {'general', 'crypto', 'nft', 'gaming', 'finance', 'tech', 'Other'}
        if category not in allowed_categories:
            return json_response(False, error='invalid category', status=400)
        try:
            price = float(data.get('price', data.get('price_per_post', 0)) or 0)
        except (TypeError, ValueError):
            return json_response(False, error='price must be a valid number', status=400)

        result = run_coroutine_safely(
            verify_and_register_channel(
                bot_instance.application.bot,
                username,
                owner_id,
                category,
                price,
                owner_wallet
            )
        )

        if result.get('success'):
            return json_response(True, data={'channel': result.get('channel')})

        error_msg = result.get('error') or 'Channel registration failed'
        status_code = 400
        if error_msg == 'User not found':
            status_code = 404
        return json_response(False, error=error_msg, status=status_code)

    except Exception as e:
        logger.error(f"Channel register error: {e}", exc_info=True)
        return json_response(False, error=str(e), status=500)


# -----------------------------------------------------------------------------
# CAMPAIGNS API
# -----------------------------------------------------------------------------

@flask_app.route('/api/campaign/create', methods=['POST'])
@rate_limit()
def api_create_campaign():
    try:
        data = request.get_json() or {}

        user_id_raw = data.get('user_id')
        title = str(data.get('title') or '').strip()
        text = str(data.get('text') or data.get('description') or '').strip()
        budget_raw = data.get('budget')

        if user_id_raw is None:
            return jsonify({'success': False, 'error': 'user_id: is required'}), 400

        try:
            advertiser_id = int(user_id_raw)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'user_id: must be an integer database id'}), 400

        if advertiser_id <= 0:
            return jsonify({'success': False, 'error': 'user_id: must be greater than 0'}), 400

        if not title:
            return jsonify({'success': False, 'error': 'title: is required'}), 400

        if len(title) < 3:
            return jsonify({'success': False, 'error': 'title: must be at least 3 characters'}), 400

        if not text:
            return jsonify({'success': False, 'error': 'text: is required'}), 400

        if len(text) < 10:
            return jsonify({'success': False, 'error': 'text: must be at least 10 characters'}), 400

        try:
            budget = float(budget_raw)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'budget: must be a valid number'}), 400

        if budget <= 0:
            return jsonify({'success': False, 'error': 'budget: must be greater than 0'}), 400

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE id = ?', (advertiser_id,))
            if not cursor.fetchone():
                return jsonify({'success': False, 'error': 'user_id: user not found'}), 404

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO campaigns (advertiser_id, title, text, budget, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (advertiser_id, title, text, budget, 'pending'))
            conn.commit()

            campaign_id = cursor.lastrowid
            logger.info(f"API: Created campaign {campaign_id} - {title}")

            return jsonify({
                'success': True,
                'campaign_id': campaign_id,
                'message': 'Campaign created',
                'data': {
                    'campaign': {
                        'id': campaign_id,
                        'advertiser_id': advertiser_id,
                        'title': title,
                        'text': text,
                        'budget': budget,
                        'status': 'pending'
                    }
                }
            }), 200

    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        return jsonify({'success': False, 'error': f'campaign_create: {str(e)}'}), 500


# -----------------------------------------------------------------------------
# DEALS API
# -----------------------------------------------------------------------------

@flask_app.route('/api/deals', methods=['GET'])
@rate_limit()
def api_get_deals():
    try:
        telegram_id_raw = request.args.get('user_id')
        if telegram_id_raw is None:
            return json_response(False, error='user_id (telegram id) is required', status=400)

        try:
            telegram_id = int(telegram_id_raw)
        except (TypeError, ValueError):
            return json_response(False, error='user_id must be a valid telegram id integer', status=400)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
            user_row = cursor.fetchone()
            if not user_row:
                return json_response(False, error='User not found', status=404)

            db_user_id = user_row['id']

            cursor.execute('''
                SELECT d.id, d.campaign_id, d.channel_id, d.status, d.escrow_amount, d.created_at,
                       d.media_type, d.media_url,
                       camp.advertiser_id, c.owner_id,
                       c.username as channel_handle, c.name as channel_name,
                       camp.title as campaign_title,
                       CASE
                           WHEN c.owner_id = ? THEN 'owner'
                           WHEN camp.advertiser_id = ? THEN 'advertiser'
                           ELSE 'viewer'
                       END AS role
                FROM deals d
                LEFT JOIN channels c ON d.channel_id = c.id
                LEFT JOIN campaigns camp ON d.campaign_id = camp.id
                WHERE camp.advertiser_id = ? OR c.owner_id = ?
                ORDER BY d.created_at DESC
            ''', (db_user_id, db_user_id, db_user_id, db_user_id))
            rows = cursor.fetchall()

            deals = []
            for row in rows:
                state = row['status']
                if row['advertiser_id'] == db_user_id:
                    role = 'advertiser'
                elif row['owner_id'] == db_user_id:
                    role = 'owner'
                else:
                    role = 'viewer'

                allowed = get_role_allowed_actions(state, role)

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
                    'role': role,
                    'step': DealStateMachine.get_step(state),
                    'is_terminal': DealStateMachine.is_terminal(state),
                    'allowed_transitions': allowed,
                    'created_at': row['created_at'],
                    'media_type': row['media_type'],
                    'media_url': row['media_url']
                })

            return json_response(True, data={'deals': deals})

    except Exception as e:
        logger.error(f"Error getting deals: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/leaderboard/monthly', methods=['GET'])
@rate_limit()
def api_get_monthly_leaderboard():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    c.id AS channel_id,
                    c.username,
                    c.name,
                    SUM(d.escrow_amount) AS total_earned,
                    COUNT(d.id) AS completed_count,
                    c.total_deals,
                    c.completed_deals
                FROM deals d
                JOIN channels c ON d.channel_id = c.id
                WHERE d.status = 'completed'
                  AND strftime('%Y-%m', d.created_at) = strftime('%Y-%m', 'now')
                GROUP BY c.id, c.username, c.name, c.total_deals, c.completed_deals
                ORDER BY total_earned DESC
                LIMIT 5
            ''')
            rows = cursor.fetchall()

        leaders = []
        for index, row in enumerate(rows):
            total_deals = row['total_deals'] or 0
            completed_deals = row['completed_deals'] or 0
            success_rate = 0.0
            if total_deals > 0:
                success_rate = round((completed_deals / total_deals) * 100, 2)

            leaders.append({
                'rank': index + 1,
                'channel_id': row['channel_id'],
                'username': row['username'],
                'name': row['name'],
                'total_earned': row['total_earned'] or 0,
                'completed_count': row['completed_count'] or 0,
                'success_rate': success_rate
            })

        return json_response(True, data={'leaders': leaders})

    except Exception as e:
        logger.error(f"Error getting monthly leaderboard: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>', methods=['GET'])
@rate_limit()
def api_get_single_deal(deal_id):
    deal = get_deal_with_state_info(deal_id)
    if deal:
        return json_response(True, data={'deal': deal})
    return json_response(False, error='Deal not found', status=404)


@flask_app.route('/api/deal/create', methods=['POST'])
@rate_limit()
def api_create_deal():
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        data = request.get_json() or {}

        campaign_id = data.get('campaign_id')
        channel_id = data.get('channel_id')
        advertiser_telegram_id = data.get('user_id')
        memo = data.get('memo')
        amount_raw = data.get('amount')
        media_type = data.get('media_type')
        media_url = data.get('media_url')

        missing_fields = [
            field_name for field_name, value in {
                'campaign_id': campaign_id,
                'channel_id': channel_id,
                'user_id': advertiser_telegram_id,
                'amount': amount_raw,
                'memo': memo
            }.items() if value in (None, '')
        ]
        if missing_fields:
            return json_response(False, error=f"Missing required fields: {', '.join(missing_fields)}", status=400)

        try:
            amount = float(amount_raw)
        except (TypeError, ValueError):
            return json_response(False, error='amount must be a valid number', status=400)

        try:
            campaign_id = int(campaign_id)
        except (TypeError, ValueError):
            return json_response(False, error='campaign_id must be a valid integer', status=400)

        try:
            channel_id = int(channel_id)
        except (TypeError, ValueError):
            return json_response(False, error='channel_id must be a valid integer', status=400)

        try:
            advertiser_telegram_id = int(advertiser_telegram_id)
        except (TypeError, ValueError):
            return json_response(False, error='user_id must be a valid telegram id integer', status=400)

        if campaign_id <= 0:
            return json_response(False, error='campaign_id is required', status=400)
        if channel_id <= 0:
            return json_response(False, error='channel_id is required', status=400)
        if advertiser_telegram_id <= 0:
            return json_response(False, error='user_id must be greater than 0', status=400)
        if amount <= 0:
            return json_response(False, error='amount must be greater than 0', status=400)
        if not memo:
            return json_response(False, error='memo is required', status=400)

        # Verify user, campaign and channel exist
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (advertiser_telegram_id,))
            advertiser_row = cursor.fetchone()
            if not advertiser_row:
                return json_response(False, error='User not found', status=404)
            advertiser_id = advertiser_row['id']

            cursor.execute('SELECT id FROM campaigns WHERE id = ? AND advertiser_id = ?', (campaign_id, advertiser_id))
            if not cursor.fetchone():
                return json_response(False, error='Campaign not found', status=404)

            cursor.execute('SELECT id FROM channels WHERE id = ?', (channel_id,))
            if not cursor.fetchone():
                return json_response(False, error='Channel not found', status=404)

        # Check for existing active deal (race condition prevention)
        with get_db() as conn:
            conn.execute('BEGIN IMMEDIATE')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM deals
                WHERE campaign_id = ? AND channel_id = ?
                  AND status NOT IN ('completed', 'cancelled', 'refunded')
            ''', (campaign_id, channel_id))
            existing = cursor.fetchone()
            if existing:
                conn.rollback()
                return json_response(False, error='An active deal already exists for this campaign and channel', status=409)

            cursor.execute('''
                INSERT INTO deals (campaign_id, channel_id, status, escrow_amount)
                VALUES (?, ?, ?, ?)
            ''', (campaign_id, channel_id, 'pending', amount))
            deal_id = cursor.lastrowid

            cursor.execute('''
                UPDATE deals
                   SET media_type = ?, media_url = ?
                 WHERE id = ?
            ''', (media_type, media_url, deal_id))

            cursor.execute('''
                UPDATE channels
                SET total_deals = COALESCE(total_deals, 0) + 1
                WHERE id = ?
            ''', (channel_id,))

            cursor.execute('''
                SELECT d.id, d.campaign_id, d.channel_id, d.status, d.escrow_amount, d.created_at,
                       d.media_type, d.media_url,
                       c.username as channel_handle, c.name as channel_name,
                       camp.title as campaign_title
                FROM deals d
                LEFT JOIN channels c ON d.channel_id = c.id
                LEFT JOIN campaigns camp ON d.campaign_id = camp.id
                WHERE d.id = ?
            ''', (deal_id,))
            deal_row = cursor.fetchone()

            conn.commit()

        logger.info(f"API: Created deal {deal_id}")

        if bot_instance is not None and bot_instance.application is not None and NOTIFICATIONS_AVAILABLE:
            dispatch_background_async(
                send_deal_notification(
                    bot_instance.application.bot,
                    deal_id,
                    'created',
                    {'memo': memo}
                ),
                task_name=f"deal-notify-{deal_id}-created"
            )

        return json_response(True, data={
            'status': 'pending',
            'deal_id': deal_id,
            'deal': {
                'id': deal_row['id'],
                'campaign_id': deal_row['campaign_id'],
                'channel_id': deal_row['channel_id'],
                'status': deal_row['status'],
                'label': DealStateMachine.get_label(deal_row['status']),
                'escrow_amount': deal_row['escrow_amount'],
                'amount': deal_row['escrow_amount'],
                'title': deal_row['campaign_title'] or deal_row['channel_name'] or f"Deal #{deal_row['id']}",
                'channel': deal_row['channel_handle'],
                'type': 'deal',
                'step': DealStateMachine.get_step(deal_row['status']),
                'is_terminal': DealStateMachine.is_terminal(deal_row['status']),
                'allowed_transitions': DealStateMachine.get_allowed_transitions(deal_row['status']),
                'created_at': deal_row['created_at'],
                'media_type': media_type,
                'media_url': media_url
            }
        })

    except Exception as e:
        logger.error(f"Error creating deal: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/status', methods=['POST'])
@rate_limit()
def api_update_deal_status(deal_id):
    try:
        data = request.get_json() or {}
        new_status = data.get('status', '')
        telegram_id = data.get('telegram_id')

        if not new_status:
            return json_response(False, error='status is required', status=400)

        result = transition_deal_state(deal_id, new_status, telegram_id)

        if result['success']:
            return json_response(True, data={
                'deal': result['deal'],
                'old_status': result['old_state'],
                'new_status': result['new_state']
            })
        else:
            status_code = 409 if result.get('conflict') else 400
            return json_response(False, error=result['error'], status=status_code)

    except Exception as e:
        logger.error(f"Error updating deal: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/transition', methods=['POST'])
@rate_limit()
def api_transition_deal(deal_id):
    try:
        data = request.get_json() or {}
        new_state = data.get('state') or data.get('status')
        telegram_id = data.get('telegram_id')

        if not new_state:
            return json_response(False, error='state is required', status=400)

        # Permission check: user must have appropriate role for this channel
        if not telegram_id:
            return json_response(False, error='telegram_id is required for permission check', status=400)

        # Get deal info to know channel_id
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT channel_id, status FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()
            if not deal:
                return json_response(False, error='Deal not found', status=404)

            channel_id = deal['channel_id']
            user_id = get_user_id(telegram_id)

        # Map new_state to action for permission check
        action_map = {
            'accepted': 'accept_deal',
            'posted': 'post_ad',
            'completed': 'release_escrow'
        }
        action = action_map.get(new_state)
        if action:
            perm = check_channel_permission(user_id, channel_id, action)
            if not perm['allowed']:
                return json_response(False, error=perm['error'], status=403)

        # Proceed with transition
        result = transition_deal_state(deal_id, new_state, telegram_id)

        if result['success']:
            # Send notification asynchronously (fire and forget)
            if bot_instance and bot_instance.application:
                try:
                    dispatch_background_async(
                        send_deal_notification(
                            bot_instance.application.bot,
                            deal_id,
                            new_state
                        ),
                        task_name=f"deal-notify-{deal_id}-{new_state}"
                    )
                except Exception as notif_err:
                    logger.warning(f"Notification error: {notif_err}")

            return json_response(True, data={
                'deal': result['deal'],
                'transition': f"{result['old_state']} → {result['new_state']}"
            })
        else:
            status_code = 409 if result.get('conflict') else 400
            return json_response(False, error=result['error'], status=status_code)

    except Exception as e:
        logger.error(f"Error transitioning deal: {e}")
        return json_response(False, error=str(e), status=500)


# -----------------------------------------------------------------------------
# PERMISSION-PROTECTED DEAL ACTIONS
# -----------------------------------------------------------------------------

@flask_app.route('/api/deal/<int:deal_id>/accept', methods=['POST'])
@rate_limit()
def api_accept_deal(deal_id):
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')

        if not telegram_id:
            return json_response(False, error='telegram_id is required', status=400)

        # Get deal info
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT channel_id, status FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()

            if not deal:
                return json_response(False, error='Deal not found', status=404)

            user_id = get_user_id(telegram_id)
            channel_id = deal['channel_id']

        # Permission check
        permission = check_channel_permission(user_id, channel_id, 'accept_deal')
        if not permission['allowed']:
            return json_response(False, error=permission['error'], status=403)

        # Update status using state machine
        result = transition_deal_state(deal_id, 'accepted', telegram_id)
        if not result['success']:
            status_code = 409 if result.get('conflict') else 400
            return json_response(False, error=result['error'], status=status_code)

        logger.info(f"Deal {deal_id} accepted by user {user_id} (role: {permission['role']})")

        return json_response(True, data={
            'deal_id': deal_id,
            'status': 'accepted',
            'role': permission['role']
        })

    except Exception as e:
        logger.error(f"Error accepting deal: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/post', methods=['POST'])
@rate_limit()
def api_post_ad(deal_id):
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')

        if not telegram_id:
            return json_response(False, error='telegram_id is required', status=400)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT channel_id, status FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()

            if not deal:
                return json_response(False, error='Deal not found', status=404)

            if deal['status'] not in ['accepted', 'funded', 'escrow']:
                return json_response(False, error='Deal must be accepted or funded before posting', status=400)

            user_id = get_user_id(telegram_id)
            channel_id = deal['channel_id']

        permission = check_channel_permission(user_id, channel_id, 'post_ad')
        if not permission['allowed']:
            return json_response(False, error=permission['error'], status=403)

        result = transition_deal_state(deal_id, 'posted', telegram_id)
        if not result['success']:
            status_code = 409 if result.get('conflict') else 400
            return json_response(False, error=result['error'], status=status_code)

        logger.info(f"Deal {deal_id} marked as posted by user {user_id} (role: {permission['role']})")

        return json_response(True, data={
            'deal_id': deal_id,
            'status': 'posted',
            'role': permission['role']
        })

    except Exception as e:
        logger.error(f"Error posting ad: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/release', methods=['POST'])
@rate_limit()
def api_release_escrow(deal_id):
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')

        if not telegram_id:
            return json_response(False, error='telegram_id is required', status=400)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT channel_id, status FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()

            if not deal:
                return json_response(False, error='Deal not found', status=404)

            if deal['status'] not in ['posted', 'verified']:
                return json_response(False, error='Deal must be posted or verified before escrow release', status=400)

            user_id = get_user_id(telegram_id)
            channel_id = deal['channel_id']

        permission = check_channel_permission(user_id, channel_id, 'release_escrow')
        if not permission['allowed']:
            return json_response(False, error=permission['error'], status=403)

        result = transition_deal_state(deal_id, 'completed', telegram_id)
        if not result['success']:
            status_code = 409 if result.get('conflict') else 400
            return json_response(False, error=result['error'], status=status_code)

        logger.info(f"Deal {deal_id} escrow released by user {user_id} (role: {permission['role']})")

        return json_response(True, data={
            'deal_id': deal_id,
            'status': 'completed',
            'role': permission['role']
        })

    except Exception as e:
        logger.error(f"Error releasing escrow: {e}")
        return json_response(False, error=str(e), status=500)



@flask_app.route('/api/admin/release/<deal_id>', methods=['POST'])
def admin_release(deal_id):
    try:
        if not supabase:
            return {"success": False, "error": "Supabase is not configured"}, 503

        if not sync_send_from_platform_wallet:
            return {"success": False, "error": "TON escrow module not available"}, 503

        data = request.get_json(silent=True) or {}
        force_override = bool(data.get("force_override"))

        deal_resp = supabase.table("deals").select("*").eq("id", deal_id).single().execute()
        if not deal_resp.data:
            return {"success": False, "error": "Deal not found"}, 404

        deal = deal_resp.data

        if deal.get("status") in ["released", "refunded"]:
            return {"success": False, "error": "Deal already finalized"}, 400

        if deal.get("status") != "ad_posted" and not force_override:
            return {"success": False, "error": "Deal must be ad_posted unless force_override is true"}, 400

        channel_resp = supabase.table("channels").select("owner_wallet").eq("id", deal["channel_id"]).single().execute()
        if not channel_resp.data:
            return {"success": False, "error": "Channel not found"}, 404

        owner_wallet = channel_resp.data["owner_wallet"]
        amount = float(deal.get("amount") or deal.get("escrow_amount") or 0)

        result = sync_send_from_platform_wallet(owner_wallet, amount)

        if not result["success"]:
            return {"success": False, "error": result["error"]}, 500

        supabase.table("deals").update({
            "status": "released",
            "released_at": datetime.utcnow().isoformat(),
            "escrow_tx_hash": result["tx_hash"]
        }).eq("id", deal_id).execute()

        return {"success": True, "tx_hash": result["tx_hash"]}

    except Exception as e:
        return {"success": False, "error": str(e)}, 500



@flask_app.route('/api/deal/<int:deal_id>/ad-posted', methods=['POST'])
@rate_limit()
def api_mark_ad_posted(deal_id):
    try:
        if not supabase:
            return {"success": False, "error": "Supabase is not configured"}, 503

        deal_resp = supabase.table("deals").select("*").eq("id", deal_id).single().execute()
        if not deal_resp.data:
            return {"success": False, "error": "Deal not found"}, 404

        deal = deal_resp.data
        if deal.get("status") != "paid":
            return {"success": False, "error": "Deal must be in paid state"}, 400

        now = datetime.utcnow()
        supabase.table("deals").update({
            "status": "ad_posted",
            "ad_posted_at": now.isoformat(),
            "release_at": (now + timedelta(hours=24)).isoformat()
        }).eq("id", deal_id).execute()

        return {"success": True, "deal_id": deal_id, "status": "ad_posted"}

    except Exception as e:
        return {"success": False, "error": str(e)}, 500


@flask_app.route('/api/deal/action', methods=['POST'])
@rate_limit()
def api_deal_action():
    try:
        data = request.get_json() or {}

        deal_id = data.get('deal_id')
        action = data.get('action')
        telegram_id = data.get('user_id')

        if deal_id in (None, ''):
            return json_response(False, error='deal_id is required', status=400)
        if not action:
            return json_response(False, error='action is required', status=400)
        if telegram_id in (None, ''):
            return json_response(False, error='user_id is required', status=400)

        try:
            deal_id = int(deal_id)
        except (TypeError, ValueError):
            return json_response(False, error='deal_id must be a valid integer', status=400)

        try:
            telegram_id = int(telegram_id)
        except (TypeError, ValueError):
            return json_response(False, error='user_id must be a valid telegram id integer', status=400)

        target_state = DEAL_ACTION_TO_STATE.get(action)
        if not target_state:
            return json_response(False, error='Invalid action', status=400)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE telegram_id = ?', (telegram_id,))
            user_row = cursor.fetchone()
            if not user_row:
                return json_response(False, error='User not found', status=404)
            db_user_id = user_row['id']

            cursor.execute('''
                SELECT d.status, camp.advertiser_id, c.owner_id
                FROM deals d
                LEFT JOIN campaigns camp ON d.campaign_id = camp.id
                LEFT JOIN channels c ON d.channel_id = c.id
                WHERE d.id = ?
            ''', (deal_id,))
            deal_row = cursor.fetchone()
            if not deal_row:
                return json_response(False, error='Deal not found', status=404)

            advertiser_id = deal_row['advertiser_id']
            owner_id = deal_row['owner_id']
            current_status = deal_row['status']

        if advertiser_id == db_user_id:
            role = 'advertiser'
        elif owner_id == db_user_id:
            role = 'owner'
        else:
            return json_response(False, error='Not authorized', status=403)

        if action == 'fund' and role != 'advertiser':
            return json_response(False, error='Not authorized', status=403)

        if action in ['accept', 'reject', 'mark_posted', 'delete'] and role != 'owner':
            return json_response(False, error='Not authorized', status=403)

        if action in ['verify', 'cancel', 'dispute'] and role != 'advertiser':
            return json_response(False, error='Not authorized', status=403)

        allowed_actions = get_role_allowed_actions(current_status, role)
        if action not in allowed_actions:
            return json_response(False, error='Action not allowed in current state', status=409)

        result = transition_deal_state(deal_id, target_state, telegram_id)
        if not result['success']:
            status_code = 409 if result.get('conflict') else 400
            return json_response(False, error=result['error'], status=status_code)

        return json_response(True, data={
            'deal': result['deal'],
            'old_status': result['old_state'],
            'new_status': result['new_state']
        })

    except Exception as e:
        logger.error(f"Error handling deal action: {e}")
        return json_response(False, error=str(e), status=500)

# -----------------------------------------------------------------------------
# CHANNEL ADMIN MANAGEMENT API
# -----------------------------------------------------------------------------

@flask_app.route('/api/channel/<int:channel_id>/admins', methods=['GET'])
@rate_limit()
def api_get_channel_admins(channel_id):
    try:
        admins = get_channel_admins(channel_id)
        return json_response(True, data={'admins': admins})
    except Exception as e:
        logger.error(f"Error getting channel admins: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/channel/<int:channel_id>/admins', methods=['POST'])
@rate_limit()
def api_add_channel_admin(channel_id):
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        role = data.get('role', ChannelRole.POSTER)

        if not telegram_id:
            return json_response(False, error='telegram_id is required', status=400)

        if role not in [ChannelRole.OWNER, ChannelRole.MANAGER, ChannelRole.POSTER]:
            return json_response(False, error='Invalid role. Must be: owner, manager, or poster', status=400)

        user_id = get_user_id(telegram_id)

        success = set_channel_admin(channel_id, user_id, role)

        if success:
            return json_response(True, data={
                'channel_id': channel_id,
                'user_id': user_id,
                'role': role
            })
        else:
            return json_response(False, error='Failed to add admin', status=500)

    except Exception as e:
        logger.error(f"Error adding channel admin: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/channel/<int:channel_id>/admins/<int:user_id>', methods=['DELETE'])
@rate_limit()
def api_remove_channel_admin(channel_id, user_id):
    try:
        success = remove_channel_admin(channel_id, user_id)
        if success:
            return json_response(True, data={'message': 'Admin removed'})
        else:
            return json_response(False, error='Admin not found', status=404)

    except Exception as e:
        logger.error(f"Error removing channel admin: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/channel/<int:channel_id>/verify', methods=['POST'])
@rate_limit()
def api_verify_channel_admin(channel_id):
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')

        if not telegram_id:
            return json_response(False, error='telegram_id is required', status=400)

        if not bot_instance:
            return json_response(False, error='Bot not available for verification', status=503)

        # This requires async execution - return instruction for bot verification
        return json_response(True, data={
            'message': 'Verification requested. Use bot command /verify to complete.',
            'channel_id': channel_id,
            'telegram_id': telegram_id
        })

    except Exception as e:
        logger.error(f"Error verifying admin: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/permission/check', methods=['POST'])
@rate_limit()
def api_check_permission():
    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        channel_id = data.get('channel_id')
        action = data.get('action')

        if not all([telegram_id, channel_id, action]):
            return json_response(False, error='telegram_id, channel_id, and action are required', status=400)

        user_id = get_user_id(telegram_id)
        result = check_channel_permission(user_id, channel_id, action)

        return json_response(True, data={
            'allowed': result['allowed'],
            'role': result['role'],
            'error': result['error']
        })

    except Exception as e:
        logger.error(f"Error checking permission: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/health')
def health_check():
    return json_response(True, data={
        'service': 'tg-adescrow-bot',
        'database': DATABASE_PATH,
        'ton_escrow': TON_ESCROW_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })


# -----------------------------------------------------------------------------
# TON ESCROW API
# -----------------------------------------------------------------------------

@flask_app.route('/api/deal/<int:deal_id>/escrow/create', methods=['POST'])
@rate_limit()
def api_create_escrow_wallet(deal_id):
    if not TON_ESCROW_AVAILABLE:
        return json_response(False, error='TON escrow module not available. Install tonsdk, cryptography, aiohttp.', status=503)

    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, status, escrow_amount FROM deals WHERE id = ?', (deal_id,))
            deal = cursor.fetchone()

            if not deal:
                return json_response(False, error='Deal not found', status=404)

            # Check if wallet already exists
            cursor.execute('SELECT address FROM escrow_wallets WHERE deal_id = ?', (deal_id,))
            existing = cursor.fetchone()
            if existing:
                return json_response(True, data={
                    'wallet': {
                        'deal_id': deal_id,
                        'address': existing['address'],
                        'expected_amount': deal['escrow_amount']
                    }
                })

            # Generate new wallet
            wallet_info = ton_escrow.generate_escrow_wallet()

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

            return json_response(True, data={'wallet': {
                'id': wallet_id,
                'deal_id': deal_id,
                'address': wallet_info['address'],
                'expected_amount': deal['escrow_amount'],
                'network': ton_escrow.TON_NETWORK
            }})

    except Exception as e:
        logger.error(f"Error creating escrow wallet: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/escrow/status', methods=['GET'])
@rate_limit()
def api_get_escrow_status(deal_id):
    if not TON_ESCROW_AVAILABLE:
        return json_response(False, error='TON escrow module not available', status=503)

    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.id, d.status, d.escrow_amount, d.advertiser_wallet, d.channel_owner_wallet,
                       w.id as wallet_id, w.address, w.balance as cached_balance, w.last_checked
                FROM deals d
                LEFT JOIN escrow_wallets w ON d.id = w.deal_id
                WHERE d.id = ?
            ''', (deal_id,))
            row = cursor.fetchone()

            if not row:
                return json_response(False, error='Deal not found', status=404)

            if not row['address']:
                return json_response(False, error='No escrow wallet created for this deal', status=404)

            # Get live balance from blockchain
            balance_info = run_async(ton_escrow.get_wallet_balance(row['address']))

            expected = row['escrow_amount'] or 0
            current_balance = balance_info.get('balance', 0)
            is_funded = current_balance >= expected * 0.99 if expected > 0 else False

            cursor.execute('''
                UPDATE escrow_wallets SET balance = ?, last_checked = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (current_balance, row['wallet_id']))
            conn.commit()

            return json_response(True, data={'escrow': {
                'deal_id': deal_id,
                'deal_status': row['status'],
                'address': row['address'],
                'expected_amount': expected,
                'current_balance': current_balance,
                'is_funded': is_funded,
                'network': ton_escrow.TON_NETWORK,
                'last_checked': datetime.now().isoformat()
            }})

    except Exception as e:
        logger.error(f"Error getting escrow status: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/escrow/verify', methods=['POST'])
@rate_limit()
def api_verify_escrow_deposit(deal_id):
    if not TON_ESCROW_AVAILABLE:
        return json_response(False, error='TON escrow module not available', status=503)

    try:
        data = request.get_json() or {}
        advertiser_wallet = data.get('advertiser_wallet')

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.id, d.status, d.escrow_amount,
                       w.id as wallet_id, w.address
                FROM deals d
                JOIN escrow_wallets w ON d.id = w.deal_id
                WHERE d.id = ?
            ''', (deal_id,))
            row = cursor.fetchone()

            if not row:
                return json_response(False, error='Deal or escrow wallet not found', status=404)

            deposit_info = run_async(ton_escrow.check_for_deposit(row['address'], row['escrow_amount']))

            if deposit_info.get('funded'):
                cursor.execute('''
                    UPDATE deals
                    SET status = 'funded', advertiser_wallet = ?, release_at = COALESCE(release_at, ?)
                    WHERE id = ? AND status IN ('pending', 'accepted')
                ''', (
                    advertiser_wallet or deposit_info.get('from_address'),
                    (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                    deal_id
                ))

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

                cursor.execute('''
                    UPDATE escrow_wallets SET balance = ?, last_checked = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (deposit_info.get('received_amount'), row['wallet_id']))

                conn.commit()
                logger.info(f"Deal {deal_id} funded with {deposit_info.get('received_amount')} TON")

                return json_response(True, data={
                    'funded': True,
                    'deal_id': deal_id,
                    'new_status': 'funded',
                    'received_amount': deposit_info.get('received_amount'),
                    'transaction_hash': deposit_info.get('transaction_hash'),
                    'from_address': deposit_info.get('from_address')
                })
            else:
                return json_response(True, data={
                    'funded': False,
                    'deal_id': deal_id,
                    'current_status': row['status'],
                    'expected_amount': row['escrow_amount'],
                    'received_amount': deposit_info.get('received_amount', 0),
                    'message': 'Deposit not detected or amount insufficient'
                })

    except Exception as e:
        logger.error(f"Error verifying deposit: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/escrow/release', methods=['POST'])
@rate_limit()
def api_release_escrow_funds(deal_id):
    if not TON_ESCROW_AVAILABLE:
        return json_response(False, error='TON escrow module not available', status=503)

    try:
        data = request.get_json() or {}
        telegram_id = data.get('telegram_id')
        channel_owner_wallet = data.get('channel_owner_wallet')

        # Permission check
        if not telegram_id:
            return json_response(False, error='telegram_id is required for permission check', status=400)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT channel_id FROM deals WHERE id = ?', (deal_id,))
            deal_row = cursor.fetchone()
            if not deal_row:
                return json_response(False, error='Deal not found', status=404)

            channel_id = deal_row['channel_id']
            user_id = get_user_id(telegram_id)

        perm = check_channel_permission(user_id, channel_id, 'release_escrow')
        if not perm['allowed']:
            return json_response(False, error=perm['error'], status=403)

        # Proceed with release
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.id, d.status, d.escrow_amount, d.channel_id, d.channel_owner_wallet,
                       w.id as wallet_id, w.address, w.encrypted_private_key, w.balance,
                       c.owner_wallet, c.owner_ton_wallet
                FROM deals d
                JOIN escrow_wallets w ON d.id = w.deal_id
                LEFT JOIN channels c ON d.channel_id = c.id
                WHERE d.id = ?
            ''', (deal_id,))
            row = cursor.fetchone()

            if not row:
                return json_response(False, error='Deal or escrow wallet not found', status=404)

            if row['status'] not in ['funded', 'posted', 'verified']:
                return json_response(False, error=f"Cannot release from status '{row['status']}'. Must be funded, posted, or verified.", status=400)

            dest_wallet = channel_owner_wallet or row['channel_owner_wallet'] or row['owner_wallet'] or row['owner_ton_wallet']
            if not dest_wallet:
                return json_response(False, error='No destination wallet specified. Provide channel_owner_wallet in request.', status=400)

            balance_info = run_async(ton_escrow.get_wallet_balance(row['address']))
            current_balance = balance_info.get('balance', 0)

            if current_balance <= MIN_ESCROW_BALANCE:
                return json_response(False, error=f'Insufficient balance: {current_balance} TON', status=400)

            release_result = run_async(
                ton_escrow.release_funds(row['encrypted_private_key'], dest_wallet, current_balance)
            )

            if release_result.get('success'):
                cursor.execute('''
                    UPDATE deals SET status = 'completed', channel_owner_wallet = ?
                    WHERE id = ?
                ''', (dest_wallet, deal_id))

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

                cursor.execute('''
                    UPDATE escrow_wallets SET balance = 0, last_checked = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (row['wallet_id'],))

                conn.commit()
                logger.info(f"Released {current_balance} TON from deal {deal_id} to {dest_wallet[:20]}...")

                return json_response(True, data={
                    'deal_id': deal_id,
                    'new_status': 'completed',
                    'released_amount': current_balance,
                    'destination': dest_wallet,
                    'tx_hash': release_result.get('tx_hash')
                })
            else:
                return json_response(False, error=release_result.get('error', 'Release transaction failed'), status=500)

    except Exception as e:
        logger.error(f"Error releasing escrow: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/escrow/refund', methods=['POST'])
@rate_limit()
def api_refund_escrow(deal_id):
    if not TON_ESCROW_AVAILABLE:
        return json_response(False, error='TON escrow module not available', status=503)

    try:
        data = request.get_json() or {}
        advertiser_wallet = data.get('advertiser_wallet')

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.id, d.status, d.escrow_amount, d.advertiser_wallet,
                       w.id as wallet_id, w.address, w.encrypted_private_key, w.balance
                FROM deals d
                JOIN escrow_wallets w ON d.id = w.deal_id
                WHERE d.id = ?
            ''', (deal_id,))
            row = cursor.fetchone()

            if not row:
                return json_response(False, error='Deal or escrow wallet not found', status=404)

            if row['status'] not in ['funded', 'posted', 'verified']:
                return json_response(False, error=f"Cannot refund from status '{row['status']}'. Must be funded, posted, or verified.", status=400)

            dest_wallet = advertiser_wallet or row['advertiser_wallet']
            if not dest_wallet:
                return json_response(False, error='No advertiser wallet known. Provide advertiser_wallet in request.', status=400)

            balance_info = run_async(ton_escrow.get_wallet_balance(row['address']))
            current_balance = balance_info.get('balance', 0)

            if current_balance <= MIN_ESCROW_BALANCE:
                return json_response(False, error=f'Insufficient balance for refund: {current_balance} TON', status=400)

            refund_result = run_async(
                ton_escrow.refund_funds(row['encrypted_private_key'], dest_wallet, current_balance)
            )

            if refund_result.get('success'):
                cursor.execute('UPDATE deals SET status = ? WHERE id = ?', ('refunded', deal_id))

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

                cursor.execute('''
                    UPDATE escrow_wallets SET balance = 0, last_checked = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (row['wallet_id'],))

                conn.commit()
                logger.info(f"Refunded {current_balance} TON from deal {deal_id} to {dest_wallet[:20]}...")

                return json_response(True, data={
                    'deal_id': deal_id,
                    'new_status': 'refunded',
                    'refunded_amount': current_balance,
                    'destination': dest_wallet,
                    'tx_hash': refund_result.get('tx_hash')
                })
            else:
                return json_response(False, error=refund_result.get('error', 'Refund transaction failed'), status=500)

    except Exception as e:
        logger.error(f"Error refunding escrow: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/escrow/transactions', methods=['GET'])
@rate_limit()
def api_get_escrow_transactions(deal_id):
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

            return json_response(True, data={
                'deal_id': deal_id,
                'transactions': transactions
            })

    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return json_response(False, error=str(e), status=500)


# -----------------------------------------------------------------------------
# AUTO-POSTER API
# -----------------------------------------------------------------------------

@flask_app.route('/api/deal/<int:deal_id>/post/schedule', methods=['POST'])
@rate_limit()
def api_schedule_post(deal_id):
    if not AUTO_POSTER_AVAILABLE:
        return json_response(False, error='Auto-poster module not available', status=503)

    try:
        data = request.get_json() or {}
        scheduled_time_str = data.get('scheduled_time')
        ad_text = data.get('ad_text')
        hold_hours = data.get('hold_hours', 24)

        if not scheduled_time_str:
            return json_response(False, error='scheduled_time is required', status=400)
        if not ad_text:
            return json_response(False, error='ad_text is required', status=400)

        try:
            scheduled_time = datetime.fromisoformat(scheduled_time_str.replace('Z', '+00:00'))
        except Exception:
            return json_response(False, error='Invalid scheduled_time format', status=400)

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
                return json_response(False, error='Deal not found', status=404)

            if deal['status'] not in ['funded', 'accepted']:
                return json_response(False, error=f"Cannot schedule from status '{deal['status']}'. Deal must be funded.", status=400)

            if not deal['bot_can_post']:
                return json_response(False, error='Bot cannot post to this channel. Verify bot is admin with posting rights.', status=400)

        result = auto_poster.schedule_post(
            deal_id=deal_id,
            channel_id=deal['channel_id'],
            ad_text=ad_text,
            scheduled_time=scheduled_time,
            hold_hours=hold_hours
        )

        if result['success']:
            return json_response(True, data={
                'post_id': result['post_id'],
                'scheduled_time': scheduled_time_str,
                'hold_hours': hold_hours,
                'message': f'Post scheduled for {scheduled_time_str}'
            })
        else:
            return json_response(False, error=result['error'], status=400)

    except Exception as e:
        logger.error(f"Error scheduling post: {e}")
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/post/now', methods=['POST'])
@rate_limit()
def api_post_now(deal_id):
    if not AUTO_POSTER_AVAILABLE:
        return json_response(False, error='Auto-poster module not available', status=503)

    try:
        data = request.get_json() or {}
        ad_text = data.get('ad_text')
        hold_hours = data.get('hold_hours', 24)

        if not ad_text:
            return json_response(False, error='ad_text is required', status=400)

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
                return json_response(False, error='Deal not found', status=404)

            if not deal['telegram_channel_id']:
                return json_response(False, error='Channel not verified', status=400)

            if not deal['bot_can_post']:
                return json_response(False, error='Bot cannot post to channel', status=400)

        result = run_async(
            auto_poster.post_to_channel(bot_instance.app.bot, deal['telegram_channel_id'], ad_text)
        )

        if result['success']:
            now = datetime.now()
            release_at = now + timedelta(hours=hold_hours)

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
                    UPDATE deals
                    SET posted_message_id = ?, posted_chat_id = ?, status = 'posted'
                    WHERE id = ?
                ''', (
                    result.get('message_id'),
                    result.get('chat_id'),
                    deal_id
                ))

                conn.commit()

            return json_response(True, data={
                'message_id': result['message_id'],
                'posted_at': now.isoformat(),
                'release_at': release_at.isoformat(),
                'hold_hours': hold_hours
            })
        else:
            return json_response(False, error=result['error'], status=500)

    except Exception as e:
        logger.error(f"Error posting now: {e}")
        return json_response(False, error=str(e), status=500)




@flask_app.route('/api/deal/delete_post', methods=['POST'])
def api_delete_post():
    try:
        data = request.get_json() or {}
        deal_id = int(data.get('deal_id'))
        telegram_user_id = int(data.get('user_id'))

        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT d.posted_message_id,
                       d.posted_chat_id,
                       u.telegram_id as owner_telegram_id
                FROM deals d
                JOIN channels c ON d.channel_id = c.id
                JOIN users u ON c.owner_id = u.id
                WHERE d.id = ?
            """, (deal_id,))
            row = cursor.fetchone()

            if not row:
                return json_response(False, error="Deal not found", status=404)

            if row['owner_telegram_id'] != telegram_user_id:
                return json_response(False, error="Not authorized", status=403)

            if not row['posted_message_id']:
                return json_response(False, error="No post to delete", status=400)

            run_async(
                bot_instance.app.bot.delete_message(
                    chat_id=row['posted_chat_id'],
                    message_id=row['posted_message_id']
                )
            )

            cursor.execute("""
                UPDATE deals
                SET status = 'deleted'
                WHERE id = ?
            """, (deal_id,))
            conn.commit()

        return json_response(True)

    except Exception as e:
        return json_response(False, error=str(e), status=500)

@flask_app.route('/api/deal/<int:deal_id>/post/verify', methods=['GET'])
@rate_limit()
def api_verify_post(deal_id):
    if not AUTO_POSTER_AVAILABLE:
        return json_response(False, error='Auto-poster module not available', status=503)

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
                return json_response(False, error='No post found for this deal', status=404)

            if not post['message_id']:
                return json_response(True, data={
                    'status': post['status'],
                    'exists': None,
                    'message': 'Post not yet sent'
                })

        result = run_async(
            auto_poster.verify_message_exists(
                bot_instance.app.bot,
                post['telegram_channel_id'],
                post['message_id']
            )
        )

        return json_response(True, data={
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
        return json_response(False, error=str(e), status=500)


@flask_app.route('/api/deal/<int:deal_id>/post/cancel', methods=['POST'])
@rate_limit()
def api_cancel_scheduled_post(deal_id):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, status FROM scheduled_posts WHERE deal_id = ?', (deal_id,))
            post = cursor.fetchone()

            if not post:
                return json_response(False, error='No post found', status=404)

            if post['status'] != 'scheduled':
                return json_response(False, error=f"Cannot cancel post with status '{post['status']}'", status=400)

            cursor.execute('DELETE FROM scheduled_posts WHERE id = ?', (post['id'],))
            cursor.execute("UPDATE deals SET status = 'funded' WHERE id = ?", (deal_id,))
            conn.commit()

        return json_response(True, data={'message': 'Scheduled post cancelled'})

    except Exception as e:
        logger.error(f"Error cancelling post: {e}")
        return json_response(False, error=str(e), status=500)

def auto_release_worker():
    while True:
        try:
            if not supabase or not sync_send_from_platform_wallet:
                time.sleep(300)
                continue

            deals_resp = (
                supabase.table("deals")
                .select("*")
                .in_("status", ["paid", "ad_posted"])
                .execute()
            )

            for deal in deals_resp.data or []:
                if deal.get("status") in ("released", "refunded"):
                    continue

                now = datetime.utcnow()

                # Safe auto-release: only after ad_posted confirmation + release_at reached
                if deal.get("status") == "ad_posted" and deal.get("release_at"):
                    try:
                        release_time = datetime.fromisoformat(deal["release_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        release_time = None

                    if release_time and now >= release_time:
                        channel_resp = (
                            supabase.table("channels")
                            .select("owner_wallet")
                            .eq("id", deal["channel_id"])
                            .single()
                            .execute()
                        )

                        if not channel_resp.data:
                            continue

                        owner_wallet = channel_resp.data.get("owner_wallet")
                        amount = float(deal.get("amount") or deal.get("escrow_amount") or 0)

                        if not owner_wallet or amount <= 0:
                            continue

                        result = sync_send_from_platform_wallet(owner_wallet, amount)

                        if result.get("success"):
                            supabase.table("deals").update({
                                "status": "released",
                                "released_at": now.isoformat(),
                                "escrow_tx_hash": result.get("tx_hash")
                            }).eq("id", deal["id"]).in_("status", ["ad_posted"]).execute()

                # Auto-refund if ad was never confirmed posted within 48h after payment
                if deal.get("status") == "paid" and deal.get("paid_at"):
                    try:
                        paid_time = datetime.fromisoformat(deal["paid_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        paid_time = None

                    if paid_time and now >= (paid_time + timedelta(hours=48)):
                        buyer_wallet = deal.get("buyer_wallet") or deal.get("advertiser_wallet")

                        if not buyer_wallet:
                            buyer_resp = (
                                supabase.table("users")
                                .select("ton_wallet")
                                .eq("id", deal.get("advertiser_id"))
                                .single()
                                .execute()
                            )
                            if buyer_resp.data:
                                buyer_wallet = buyer_resp.data.get("ton_wallet")

                        amount = float(deal.get("amount") or deal.get("escrow_amount") or 0)

                        if not buyer_wallet or amount <= 0:
                            continue

                        result = sync_send_from_platform_wallet(buyer_wallet, amount)

                        if result.get("success"):
                            supabase.table("deals").update({
                                "status": "refunded",
                                "refunded_at": now.isoformat(),
                                "escrow_tx_hash": result.get("tx_hash")
                            }).eq("id", deal["id"]).in_("status", ["paid"]).execute()

        except Exception as e:
            print("Auto release error:", e)

        time.sleep(300)


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

    if not TOKEN:
        raise ValueError("BOT_TOKEN environment variable is required")

    if bot_instance is None:
        bot_instance = AdEscrowBot(TOKEN)

    # Start automatic escrow release worker
    threading.Thread(target=auto_release_worker, daemon=True).start()
    logger.info("Auto-release worker started")

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
