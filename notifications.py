"""
Notifications Module
====================
Handles Telegram notifications for deal events with anti-spam protection.
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from supabase import create_client

logger = logging.getLogger(__name__)

# Initialize Supabase client locally to avoid circular import with bot.py
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to init Supabase in notifications: {e}")

# Rate limiting: track last notification time per (deal_id, event_type)
_notification_cache: Dict[str, datetime] = {}
NOTIFICATION_COOLDOWN_SECONDS = 60  # Minimum seconds between same notifications


# =============================================================================
# MESSAGE TEMPLATES
# =============================================================================

TEMPLATES = {
    'created': """
🆕 *New Escrow Deal Created*

An ad campaign escrow deal has been created for *{channel}*.

💰 Escrow Amount: *{amount} TON*
📝 Status: Pending approval

_Deal #{deal_id}_
""",

    'accepted': """
✅ *Deal Accepted*

Your ad request for *{channel}* has been approved!

💰 Escrow Amount: *{amount} TON*
📝 Next Step: Fund the escrow to proceed

_Deal #{deal_id}_
""",

    'funded': """
💰 *Payment Received*

Escrow has been funded for *{channel}*.

Amount: *{amount} TON*
Status: Ready for posting

_Please post the advertisement within 24 hours._

_Deal #{deal_id}_
""",

    'scheduled': """
📅 *Ad Scheduled*

Your advertisement has been scheduled for *{channel}*.

⏰ Posting Time: {scheduled_time}
💰 Escrow: *{amount} TON*

_You will be notified when the ad is posted._

_Deal #{deal_id}_
""",

    'posted': """
📢 *Ad Posted!*

Your advertisement is now live on *{channel}*.

💰 Escrow: *{amount} TON*
⏳ Hold Period: {hold_hours} hours
📊 Status: Awaiting verification

_Funds will be released after successful verification._

_Deal #{deal_id}_
""",

    'verified': """
✔️ *Ad Verified*

Your advertisement on *{channel}* has been verified.

💰 Escrow: *{amount} TON*
📊 Status: Ready for release

_Deal #{deal_id}_
""",

    'completed': """
🎉 *Deal Completed!*

Funds have been released for your deal with *{channel}*.

💸 Released: *{amount} TON*
📊 Status: Completed

Thank you for using AdEscrow!

_Deal #{deal_id}_
""",

    'refunded': """
↩️ *Escrow Refunded*

Your escrow for *{channel}* has been refunded.

💸 Refunded: *{amount} TON*
📝 Reason: {reason}

_Deal #{deal_id}_
""",

    'cancelled': """
❌ *Deal Cancelled*

Your deal with *{channel}* has been cancelled.

💰 Amount: *{amount} TON*
📝 Status: Cancelled

_Deal #{deal_id}_
""",
}

# Notification routing: which user types should receive each notification
NOTIFICATION_ROUTING = {
    'created': ['advertiser', 'channel_owner'],
    'accepted': ['advertiser'],
    'funded': ['channel_owner'],
    'scheduled': ['advertiser', 'channel_owner'],
    'posted': ['advertiser'],
    'verified': ['advertiser', 'channel_owner'],
    'completed': ['advertiser', 'channel_owner'],
    'refunded': ['advertiser'],
    'cancelled': ['advertiser', 'channel_owner'],
}


# =============================================================================
# NOTIFICATION FUNCTIONS
# =============================================================================

def get_notification_message(event_type: str, data: Dict[str, Any]) -> Optional[str]:
    """
    Get formatted notification message for an event type.
    
    Args:
        event_type: Type of event (accepted, funded, posted, etc.)
        data: Dictionary with template variables (channel, amount, deal_id, etc.)
    
    Returns:
        Formatted message string or None if event type unknown
    """
    template = TEMPLATES.get(event_type)
    if not template:
        logger.warning(f"Unknown notification event type: {event_type}")
        return None
    
    try:
        # Provide defaults for optional fields
        defaults = {
            'channel': 'Channel',
            'amount': 0,
            'deal_id': 0,
            'hold_hours': 24,
            'scheduled_time': 'Soon',
            'reason': 'Advertisement removed or policy violation',
        }
        merged = {**defaults, **data}
        return template.format(**merged).strip()
    except KeyError as e:
        logger.error(f"Missing template variable for {event_type}: {e}")
        return None


def should_send_notification(deal_id: int, event_type: str) -> bool:
    """
    Check if notification should be sent (anti-spam).
    
    Returns True if enough time has passed since last notification
    of this type for this deal.
    """
    cache_key = f"{deal_id}:{event_type}"
    last_sent = _notification_cache.get(cache_key)
    
    if last_sent:
        elapsed = (datetime.now() - last_sent).total_seconds()
        if elapsed < NOTIFICATION_COOLDOWN_SECONDS:
            logger.debug(f"Notification throttled: {cache_key} ({elapsed:.0f}s ago)")
            return False
    
    return True


def mark_notification_sent(deal_id: int, event_type: str):
    """Mark that a notification was sent for anti-spam tracking."""
    cache_key = f"{deal_id}:{event_type}"
    _notification_cache[cache_key] = datetime.now()


async def send_notification(
    bot,
    telegram_id: int,
    event_type: str,
    data: Dict[str, Any],
    force: bool = False
) -> Dict[str, Any]:
    """
    Send a notification to a user via Telegram.
    
    Args:
        bot: Telegram bot instance
        telegram_id: User's Telegram ID
        event_type: Type of event
        data: Template variables
        force: If True, bypass anti-spam check
    
    Returns:
        Dict with 'success', 'message_id', 'error'
    """
    result = {'success': False, 'message_id': None, 'error': None}
    
    deal_id = data.get('deal_id', 0)
    
    # Anti-spam check
    if not force and not should_send_notification(deal_id, event_type):
        result['error'] = 'Notification throttled (anti-spam)'
        return result
    
    # Get message text
    message = get_notification_message(event_type, data)
    if not message:
        result['error'] = f'Unknown event type: {event_type}'
        return result
    
    try:
        sent_msg = await bot.send_message(
            chat_id=telegram_id,
            text=message,
            parse_mode='Markdown'
        )
        
        result['success'] = True
        result['message_id'] = sent_msg.message_id
        
        # Mark as sent for anti-spam
        mark_notification_sent(deal_id, event_type)
        
        logger.info(f"Sent {event_type} notification to {telegram_id} for deal {deal_id}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Failed to send notification to {telegram_id}: {e}")
    
    return result


async def notify_deal_participants(
    bot,
    event_type: str,
    data: Dict[str, Any],
    advertiser_telegram_id: int = None,
    channel_owner_telegram_id: int = None
) -> Dict[str, Any]:
    """
    Send notifications to appropriate deal participants based on event type.
    
    Args:
        bot: Telegram bot instance
        event_type: Type of event
        data: Template variables (must include deal_id, channel, amount)
        advertiser_telegram_id: Advertiser's Telegram ID
        channel_owner_telegram_id: Channel owner's Telegram ID
    
    Returns:
        Dict with 'success', 'notifications_sent', 'errors'
    """
    result = {'success': True, 'notifications_sent': 0, 'errors': []}
    
    routing = NOTIFICATION_ROUTING.get(event_type, [])
    
    for recipient_type in routing:
        telegram_id = None
        
        if recipient_type == 'advertiser' and advertiser_telegram_id:
            telegram_id = advertiser_telegram_id
        elif recipient_type == 'channel_owner' and channel_owner_telegram_id:
            telegram_id = channel_owner_telegram_id
        
        if telegram_id:
            send_result = await send_notification(bot, telegram_id, event_type, data)
            
            if send_result['success']:
                result['notifications_sent'] += 1
            else:
                result['errors'].append({
                    'recipient': recipient_type,
                    'telegram_id': telegram_id,
                    'error': send_result['error']
                })
    
    if result['errors']:
        result['success'] = False
    
    return result


def get_deal_notification_data(deal_row: dict, channel_row: dict = None) -> Dict[str, Any]:
    """
    Extract notification data from database rows.
    
    Args:
        deal_row: Deal database row (dict-like)
        channel_row: Channel database row (optional)
    
    Returns:
        Dict with template variables
    """
    data = {
        'deal_id': deal_row.get('id', 0),
        'amount': deal_row.get('escrow_amount', 0),
        'channel': (
            channel_row.get('username') if channel_row
            else deal_row.get('channel_handle', 'Channel')
        ),
    }
    
    # Add optional fields if present
    if deal_row.get('hold_hours'):
        data['hold_hours'] = deal_row['hold_hours']
    
    return data


# =============================================================================
# HELPER FOR GETTING PARTICIPANT IDS
# =============================================================================

def get_deal_participants(deal_id: int) -> Dict[str, Optional[int]]:
    """
    Get Telegram IDs of deal participants from Supabase.
    
    Returns:
        Dict with 'advertiser_telegram_id', 'channel_owner_telegram_id'
    """
    result = {
        'advertiser_telegram_id': None,
        'channel_owner_telegram_id': None
    }
    
    if not supabase:
        return result

    try:
        # Get deal with campaign(advertiser_id) and channels(owner_id)
        # Assuming relations: deals.campaign_id->campaigns.id, deals.channel_id->channels.id
        resp = supabase.table("deals").select("""
            campaign_id, channel_id,
            campaigns(advertiser_id),
            channels(owner_id)
        """).eq("id", deal_id).single().execute()
        
        if not resp.data:
            return result
        
        deal = resp.data
        campaign_adv_id = None
        channel_owner_id = None
        
        if deal.get('campaigns'):
            campaign_adv_id = deal['campaigns'].get('advertiser_id')
        
        if deal.get('channels'):
            channel_owner_id = deal['channels'].get('owner_id')
            
        # Get advertiser telegram_id
        if campaign_adv_id:
            user_resp = supabase.table("app_users").select("telegram_id").eq("id", campaign_adv_id).single().execute()
            if user_resp.data:
                result['advertiser_telegram_id'] = user_resp.data['telegram_id']
                
        # Get channel owner telegram_id
        if channel_owner_id:
            user_resp = supabase.table("app_users").select("telegram_id").eq("id", channel_owner_id).single().execute()
            if user_resp.data:
                result['channel_owner_telegram_id'] = user_resp.data['telegram_id']
        
    except Exception as e:
        logger.error(f"Error getting deal participants: {e}")
    
    return result


def get_deal_data_for_notification(deal_id: int) -> Optional[Dict[str, Any]]:
    """
    Get full deal data needed for notifications from Supabase.
    
    Returns:
        Dict with deal_id, amount, channel, and participant IDs
    """
    if not supabase:
        return None

    try:
        resp = supabase.table("deals").select("""
            id, escrow_amount, hold_hours,
            channels(username, name)
        """).eq("id", deal_id).single().execute()
        
        if not resp.data:
            return None
        
        deal = resp.data
        channel = deal.get('channels') or {}
        
        participants = get_deal_participants(deal_id)
        
        return {
            'deal_id': deal['id'],
            'amount': deal.get('escrow_amount') or 0,
            'channel': channel.get('username') or channel.get('name') or 'Channel',
            'hold_hours': deal.get('hold_hours') or 24,
            **participants
        }
        
    except Exception as e:
        logger.error(f"Error getting deal data: {e}")
        return None
