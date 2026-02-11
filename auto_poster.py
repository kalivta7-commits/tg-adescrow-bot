"""
Auto-Poster Module
==================
Handles scheduled Telegram channel posting with verification and escrow integration.
"""

import os
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from threading import Thread
import time

logger = logging.getLogger(__name__)

# Import notifications module
try:
    import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

# Default hold period before escrow release (hours)
DEFAULT_HOLD_HOURS = 24

# Background task intervals (seconds)
POST_CHECK_INTERVAL = 60       # Check for due posts every minute
VERIFY_CHECK_INTERVAL = 300    # Verify existing posts every 5 minutes


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_db_path():
    """Get database path from environment or default"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')


def get_db_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# POST SCHEDULING
# =============================================================================

def schedule_post(
    deal_id: int,
    channel_id: int,
    ad_text: str,
    scheduled_time: datetime,
    hold_hours: int = DEFAULT_HOLD_HOURS
) -> Dict[str, Any]:
    """
    Schedule a post for future delivery.
    
    Args:
        deal_id: The deal this post belongs to
        channel_id: Target channel database ID
        ad_text: Content to post
        scheduled_time: When to post
        hold_hours: Hours to hold escrow after posting
    
    Returns:
        dict with 'success', 'post_id', 'error'
    """
    result = {'success': False, 'post_id': None, 'error': None}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if post already scheduled for this deal
        cursor.execute('SELECT id FROM scheduled_posts WHERE deal_id = ?', (deal_id,))
        existing = cursor.fetchone()
        
        if existing:
            result['error'] = f'Post already scheduled for deal {deal_id}'
            conn.close()
            return result
        
        # Insert scheduled post
        cursor.execute('''
            INSERT INTO scheduled_posts 
            (deal_id, channel_id, ad_text, scheduled_time, hold_hours, status)
            VALUES (?, ?, ?, ?, ?, 'scheduled')
        ''', (deal_id, channel_id, ad_text, scheduled_time.isoformat(), hold_hours))
        
        post_id = cursor.lastrowid
        
        # Update deal status
        cursor.execute(
            "UPDATE deals SET status = 'scheduled' WHERE id = ?",
            (deal_id,)
        )
        
        conn.commit()
        conn.close()
        
        result['success'] = True
        result['post_id'] = post_id
        logger.info(f"Scheduled post {post_id} for deal {deal_id} at {scheduled_time}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error scheduling post: {e}")
    
    return result


def get_pending_posts() -> List[dict]:
    """Get all scheduled posts that are due for posting"""
    posts = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        cursor.execute('''
            SELECT sp.*, c.telegram_channel_id, c.username as channel_handle
            FROM scheduled_posts sp
            JOIN channels c ON sp.channel_id = c.id
            WHERE sp.status = 'scheduled' AND sp.scheduled_time <= ?
        ''', (now,))
        
        posts = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
    except Exception as e:
        logger.error(f"Error getting pending posts: {e}")
    
    return posts


def get_posts_for_verification() -> List[dict]:
    """Get all posted ads that need verification"""
    posts = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT sp.*, c.telegram_channel_id, c.username as channel_handle,
                   ew.encrypted_private_key, ew.address as escrow_address,
                   d.channel_owner_wallet, ch.owner_ton_wallet
            FROM scheduled_posts sp
            JOIN channels c ON sp.channel_id = c.id
            JOIN deals d ON sp.deal_id = d.id
            LEFT JOIN escrow_wallets ew ON d.id = ew.deal_id
            LEFT JOIN channels ch ON d.channel_id = ch.id
            WHERE sp.status = 'posted' AND sp.message_id IS NOT NULL
        ''')
        
        posts = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
    except Exception as e:
        logger.error(f"Error getting posts for verification: {e}")
    
    return posts


# =============================================================================
# TELEGRAM POSTING
# =============================================================================

async def post_to_channel(bot, channel_id: int, text: str) -> Dict[str, Any]:
    """
    Post message to Telegram channel.
    
    Args:
        bot: Telegram bot instance
        channel_id: Telegram channel ID (numeric)
        text: Message text
    
    Returns:
        dict with 'success', 'message_id', 'error'
    """
    result = {'success': False, 'message_id': None, 'error': None}
    
    try:
        message = await bot.send_message(
            chat_id=channel_id,
            text=text,
            parse_mode='HTML'
        )
        
        result['success'] = True
        result['message_id'] = message.message_id
        logger.info(f"Posted to channel {channel_id}, message_id={message.message_id}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error posting to channel {channel_id}: {e}")
    
    return result


async def verify_message_exists(bot, channel_id: int, message_id: int) -> Dict[str, Any]:
    """
    Verify if a message still exists in the channel.
    Uses forward trick to check - if message can be forwarded, it exists.
    
    Returns:
        dict with 'exists', 'error'
    """
    result = {'exists': False, 'error': None}
    
    try:
        # Try to get chat and forward the message to ourselves
        # Actually, we'll use getMessage API via copyMessage with dry_run concept
        # Simpler approach: try to edit message (will fail if deleted, succeed if exists)
        # But edit might change content...
        
        # Best approach: Use getMessages API or check via getChatHistory
        # For simplicity, try to forward to the bot's saved messages
        # This reveals if message exists
        
        # Actually just try to copy the message - if it fails, message is gone
        try:
            # Get bot's own chat ID for testing
            me = await bot.get_me()
            
            # Try to forward message to check if it exists
            # This will raise exception if message doesn't exist
            await bot.forward_message(
                chat_id=me.id,  # Forward to bot's own chat
                from_chat_id=channel_id,
                message_id=message_id
            )
            result['exists'] = True
            
        except Exception as inner_e:
            error_str = str(inner_e).lower()
            if 'message to forward not found' in error_str or 'message not found' in error_str:
                result['exists'] = False
            elif 'bot can\'t forward' in error_str:
                # Bot can't forward but message might exist, assume true
                result['exists'] = True
                logger.warning(f"Cannot forward to verify, assuming message exists")
            else:
                # Unknown error
                result['error'] = str(inner_e)
                logger.warning(f"Verification error: {inner_e}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error verifying message: {e}")
    
    return result


def update_post_status(
    post_id: int,
    status: str,
    message_id: int = None,
    posted_at: datetime = None,
    release_at: datetime = None
) -> bool:
    """Update scheduled post status in database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        updates = ['status = ?']
        params = [status]
        
        if message_id is not None:
            updates.append('message_id = ?')
            params.append(message_id)
        
        if posted_at is not None:
            updates.append('posted_at = ?')
            params.append(posted_at.isoformat())
        
        if release_at is not None:
            updates.append('release_at = ?')
            params.append(release_at.isoformat())
        
        updates.append('last_verified = ?')
        params.append(datetime.now().isoformat())
        
        params.append(post_id)
        
        cursor.execute(f'''
            UPDATE scheduled_posts SET {', '.join(updates)} WHERE id = ?
        ''', params)
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error updating post status: {e}")
        return False


def update_deal_posted(deal_id: int, message_id: int) -> bool:
    """Update deal with posting info"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE deals SET 
                status = 'posted',
                message_id = ?,
                posted_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (message_id, deal_id))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error updating deal posted status: {e}")
        return False


# =============================================================================
# BACKGROUND SCHEDULER
# =============================================================================

class PostScheduler:
    """Background scheduler for automatic posting and verification"""
    
    def __init__(self, bot_app):
        self.bot_app = bot_app
        self.running = False
        self._thread = None
    
    def start(self):
        """Start background scheduler"""
        if self.running:
            return
        
        self.running = True
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Post scheduler started")
    
    def stop(self):
        """Stop background scheduler"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Post scheduler stopped")
    
    def _run_loop(self):
        """Main scheduler loop"""
        post_counter = 0
        verify_counter = 0
        
        while self.running:
            try:
                post_counter += 1
                verify_counter += 1
                
                # Check for pending posts every minute
                if post_counter >= 60:
                    post_counter = 0
                    asyncio.run(self._process_pending_posts())
                
                # Verify existing posts every 5 minutes
                if verify_counter >= 300:
                    verify_counter = 0
                    asyncio.run(self._verify_posted_ads())
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5)
    
    async def _process_pending_posts(self):
        """Process all pending scheduled posts"""
        posts = get_pending_posts()
        
        for post in posts:
            try:
                channel_id = post.get('telegram_channel_id')
                if not channel_id:
                    logger.error(f"No telegram_channel_id for post {post['id']}")
                    continue
                
                # Post to channel
                bot = self.bot_app.bot
                result = await post_to_channel(bot, channel_id, post['ad_text'])
                
                if result['success']:
                    now = datetime.now()
                    release_at = now + timedelta(hours=post['hold_hours'])
                    
                    # Update post record
                    update_post_status(
                        post['id'],
                        'posted',
                        message_id=result['message_id'],
                        posted_at=now,
                        release_at=release_at
                    )
                    
                    # Update deal
                    update_deal_posted(post['deal_id'], result['message_id'])
                    
                    # Send notification to advertiser that ad was posted
                    if NOTIFICATIONS_AVAILABLE:
                        try:
                            await self._send_notification(
                                post['deal_id'],
                                'posted',
                                {'hold_hours': post['hold_hours']}
                            )
                        except Exception as notif_err:
                            logger.warning(f"Notification error: {notif_err}")
                    
                    logger.info(f"Successfully posted deal {post['deal_id']}, release at {release_at}")
                else:
                    logger.error(f"Failed to post deal {post['deal_id']}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error processing post {post['id']}: {e}")
    
    async def _verify_posted_ads(self):
        """Verify all posted ads and process escrow accordingly"""
        posts = get_posts_for_verification()
        
        for post in posts:
            try:
                channel_id = post.get('telegram_channel_id')
                message_id = post.get('message_id')
                
                if not channel_id or not message_id:
                    continue
                
                # Verify message exists
                bot = self.bot_app.bot
                result = await verify_message_exists(bot, channel_id, message_id)
                
                now = datetime.now()
                release_at = datetime.fromisoformat(post['release_at']) if post.get('release_at') else None
                
                if result['exists']:
                    # Post still exists
                    if release_at and now >= release_at:
                        # Hold period over, release escrow
                        await self._release_escrow(post)
                    else:
                        # Update last verified
                        update_post_status(post['id'], 'posted')
                        logger.debug(f"Post {post['id']} verified, waiting for release time")
                else:
                    # Post deleted, refund
                    logger.warning(f"Post {post['id']} deleted, triggering refund")
                    await self._refund_escrow(post)
                    
            except Exception as e:
                logger.error(f"Error verifying post {post['id']}: {e}")
    
    async def _send_notification(self, deal_id: int, event_type: str, extra_data: dict = None):
        """Send notification for a deal event"""
        if not NOTIFICATIONS_AVAILABLE:
            return
        
        try:
            deal_data = notifications.get_deal_data_for_notification(deal_id)
            if not deal_data:
                return
            
            if extra_data:
                deal_data.update(extra_data)
            
            bot = self.bot_app.bot
            await notifications.notify_deal_participants(
                bot=bot,
                event_type=event_type,
                data=deal_data,
                advertiser_telegram_id=deal_data.get('advertiser_telegram_id'),
                channel_owner_telegram_id=deal_data.get('channel_owner_telegram_id')
            )
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")
    
    async def _release_escrow(self, post: dict):
        """Release escrow funds to channel owner"""
        try:
            # Import here to avoid circular imports
            import ton_escrow
            
            encrypted_key = post.get('encrypted_private_key')
            dest_wallet = post.get('channel_owner_wallet') or post.get('owner_ton_wallet')
            
            if not encrypted_key or not dest_wallet:
                logger.error(f"Missing escrow info for post {post['id']}")
                update_post_status(post['id'], 'verified')
                return
            
            # Get balance
            balance = await ton_escrow.get_wallet_balance(post['escrow_address'])
            amount = balance.get('balance', 0)
            
            if amount > 0.05:
                result = await ton_escrow.release_funds(encrypted_key, dest_wallet, amount)
                
                if result['success']:
                    update_post_status(post['id'], 'released')
                    # Update deal status
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE deals SET status = 'completed' WHERE id = ?",
                        (post['deal_id'],)
                    )
                    conn.commit()
                    conn.close()
                    logger.info(f"Released escrow for post {post['id']}")
                    
                    # Send completion notification to both parties
                    if NOTIFICATIONS_AVAILABLE:
                        try:
                            await self._send_notification(post['deal_id'], 'completed')
                        except Exception as notif_err:
                            logger.warning(f"Notification error: {notif_err}")
                else:
                    logger.error(f"Escrow release failed: {result['error']}")
            else:
                # No funds to release
                update_post_status(post['id'], 'released')
                
        except Exception as e:
            logger.error(f"Error releasing escrow: {e}")
    
    async def _refund_escrow(self, post: dict):
        """Refund escrow funds to advertiser"""
        try:
            import ton_escrow
            
            encrypted_key = post.get('encrypted_private_key')
            
            if not encrypted_key:
                logger.error(f"Missing escrow key for post {post['id']}")
                update_post_status(post['id'], 'refunded')
                return
            
            # Get advertiser wallet from deal
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT advertiser_wallet FROM deals WHERE id = ?',
                (post['deal_id'],)
            )
            deal = cursor.fetchone()
            conn.close()
            
            if not deal or not deal['advertiser_wallet']:
                logger.error(f"No advertiser wallet for deal {post['deal_id']}")
                update_post_status(post['id'], 'refunded')
                return
            
            # Get balance
            balance = await ton_escrow.get_wallet_balance(post['escrow_address'])
            amount = balance.get('balance', 0)
            
            if amount > 0.05:
                result = await ton_escrow.refund_funds(
                    encrypted_key, deal['advertiser_wallet'], amount
                )
                
                if result['success']:
                    update_post_status(post['id'], 'refunded')
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE deals SET status = 'refunded' WHERE id = ?",
                        (post['deal_id'],)
                    )
                    conn.commit()
                    conn.close()
                    logger.info(f"Refunded escrow for post {post['id']}")
                else:
                    logger.error(f"Escrow refund failed: {result['error']}")
            else:
                update_post_status(post['id'], 'refunded')
                
        except Exception as e:
            logger.error(f"Error refunding escrow: {e}")


# Global scheduler instance
_scheduler: Optional[PostScheduler] = None


def start_scheduler(bot_app):
    """Start the global post scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = PostScheduler(bot_app)
    _scheduler.start()
    return _scheduler


def stop_scheduler():
    """Stop the global post scheduler"""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
        _scheduler = None
