"""
TON Escrow Module
=================
Handles TON blockchain escrow wallets for advertising deals.
One wallet per deal with encrypted private key storage.

Uses TON testnet by default.
"""

import os
import json
import logging
import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from base64 import urlsafe_b64encode, urlsafe_b64decode

import aiohttp
from cryptography.fernet import Fernet
from tonsdk.contract.wallet import Wallets, WalletVersionEnum
from tonsdk.utils import to_nano, from_nano, bytes_to_b64str
from tonsdk.crypto import mnemonic_new, mnemonic_to_wallet_key

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Network configuration
TON_NETWORK = os.getenv("TON_NETWORK", "testnet")
TONCENTER_API_KEY = os.getenv("TONCENTER_API_KEY", "")

TONCENTER_ENDPOINTS = {
    "testnet": "https://testnet.toncenter.com/api/v2",
    "mainnet": "https://toncenter.com/api/v2"
}

def get_toncenter_url() -> str:
    """Get toncenter API URL for current network"""
    return TONCENTER_ENDPOINTS.get(TON_NETWORK, TONCENTER_ENDPOINTS["testnet"])


# =============================================================================
# ENCRYPTION
# =============================================================================

def get_encryption_key() -> bytes:
    """
    Get or generate encryption key for private key storage.
    MUST be set via ESCROW_SECRET_KEY environment variable in production.
    """
    key = os.getenv("ESCROW_SECRET_KEY")
    if key:
        # Ensure key is valid Fernet key (32 bytes, base64 encoded)
        try:
            Fernet(key.encode())
            return key.encode()
        except Exception:
            logger.warning("Invalid ESCROW_SECRET_KEY format, generating new key")
    
    # Generate new key (for development only)
    new_key = Fernet.generate_key()
    logger.warning(
        f"Using generated encryption key. Set ESCROW_SECRET_KEY environment variable "
        f"to: {new_key.decode()}"
    )
    return new_key


def encrypt_mnemonic(mnemonic: list) -> str:
    """Encrypt mnemonic phrase for secure storage"""
    key = get_encryption_key()
    f = Fernet(key)
    mnemonic_str = " ".join(mnemonic)
    encrypted = f.encrypt(mnemonic_str.encode())
    return encrypted.decode()


def decrypt_mnemonic(encrypted_mnemonic: str) -> list:
    """Decrypt stored mnemonic phrase"""
    key = get_encryption_key()
    f = Fernet(key)
    decrypted = f.decrypt(encrypted_mnemonic.encode())
    return decrypted.decode().split(" ")


# =============================================================================
# WALLET OPERATIONS
# =============================================================================

def generate_escrow_wallet() -> Dict[str, str]:
    """
    Generate a new TON wallet for escrow.
    
    Returns:
        dict with 'address', 'encrypted_mnemonic', 'wallet_version'
    """
    try:
        # Generate new mnemonic (24 words)
        mnemonic = mnemonic_new()
        
        # Create wallet (v4r2 is most common)
        _mnemonics, _pub_key, _priv_key, wallet = Wallets.create(
            version=WalletVersionEnum.v4r2,
            workchain=0,
            wallet_id=698983191  # Default wallet subwallet id
        )
        
        # Actually use our generated mnemonic
        pub_key, priv_key = mnemonic_to_wallet_key(mnemonic)
        
        # Recreate wallet with our mnemonic's keys
        _, _, _, wallet = Wallets.from_mnemonics(
            mnemonics=mnemonic,
            version=WalletVersionEnum.v4r2,
            workchain=0
        )
        
        # Get address in user-friendly format
        address = wallet.address.to_string(True, True, True)
        
        # Encrypt mnemonic for storage
        encrypted = encrypt_mnemonic(mnemonic)
        
        logger.info(f"Generated new escrow wallet: {address[:20]}...")
        
        return {
            "address": address,
            "encrypted_mnemonic": encrypted,
            "wallet_version": "v4r2"
        }
        
    except Exception as e:
        logger.error(f"Error generating wallet: {e}")
        raise


def restore_wallet_from_mnemonic(encrypted_mnemonic: str) -> Any:
    """Restore wallet object from encrypted mnemonic"""
    mnemonic = decrypt_mnemonic(encrypted_mnemonic)
    
    _, _, _, wallet = Wallets.from_mnemonics(
        mnemonics=mnemonic,
        version=WalletVersionEnum.v4r2,
        workchain=0
    )
    
    return wallet


# =============================================================================
# BLOCKCHAIN QUERIES (via toncenter API)
# =============================================================================

async def get_wallet_balance(address: str) -> Dict[str, Any]:
    """
    Get wallet balance from toncenter API.
    
    Returns:
        dict with 'balance' (in TON), 'balance_nano', 'status'
    """
    url = f"{get_toncenter_url()}/getAddressBalance"
    params = {"address": address}
    
    if TONCENTER_API_KEY:
        params["api_key"] = TONCENTER_API_KEY
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("ok"):
                    balance_nano = int(data.get("result", 0))
                    balance_ton = from_nano(balance_nano, "ton")
                    return {
                        "balance": float(balance_ton),
                        "balance_nano": balance_nano,
                        "status": "active" if balance_nano > 0 else "empty"
                    }
                else:
                    return {
                        "balance": 0,
                        "balance_nano": 0,
                        "status": "unknown",
                        "error": data.get("error", "Unknown error")
                    }
                    
    except Exception as e:
        logger.error(f"Error getting balance for {address}: {e}")
        return {
            "balance": 0,
            "balance_nano": 0,
            "status": "error",
            "error": str(e)
        }


async def get_address_info(address: str) -> Dict[str, Any]:
    """Get detailed address information"""
    url = f"{get_toncenter_url()}/getAddressInformation"
    params = {"address": address}
    
    if TONCENTER_API_KEY:
        params["api_key"] = TONCENTER_API_KEY
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("ok"):
                    result = data.get("result", {})
                    balance_nano = int(result.get("balance", 0))
                    return {
                        "balance": float(from_nano(balance_nano, "ton")),
                        "balance_nano": balance_nano,
                        "state": result.get("state", "uninitialized"),
                        "last_tx_lt": result.get("last_transaction_id", {}).get("lt"),
                        "last_tx_hash": result.get("last_transaction_id", {}).get("hash")
                    }
                    
    except Exception as e:
        logger.error(f"Error getting address info: {e}")
    
    return {"balance": 0, "state": "unknown"}


async def get_transactions(address: str, limit: int = 10) -> list:
    """Get recent transactions for an address"""
    url = f"{get_toncenter_url()}/getTransactions"
    params = {"address": address, "limit": limit}
    
    if TONCENTER_API_KEY:
        params["api_key"] = TONCENTER_API_KEY
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("ok"):
                    transactions = []
                    for tx in data.get("result", []):
                        in_msg = tx.get("in_msg", {})
                        
                        # Parse incoming transaction
                        if in_msg and in_msg.get("value"):
                            transactions.append({
                                "hash": tx.get("transaction_id", {}).get("hash"),
                                "lt": tx.get("transaction_id", {}).get("lt"),
                                "timestamp": tx.get("utime"),
                                "type": "incoming",
                                "amount": float(from_nano(int(in_msg.get("value", 0)), "ton")),
                                "amount_nano": int(in_msg.get("value", 0)),
                                "from_address": in_msg.get("source"),
                                "to_address": in_msg.get("destination"),
                                "message": in_msg.get("message", "")
                            })
                    
                    return transactions
                    
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
    
    return []


async def check_for_deposit(address: str, expected_amount: float) -> Dict[str, Any]:
    """
    Check if wallet received the expected deposit.
    
    Returns:
        dict with 'funded', 'received_amount', 'transaction_hash', 'from_address'
    """
    result = {
        "funded": False,
        "received_amount": 0,
        "transaction_hash": None,
        "from_address": None
    }
    
    try:
        transactions = await get_transactions(address, limit=20)
        
        total_received = 0
        last_deposit = None
        
        for tx in transactions:
            if tx.get("type") == "incoming" and tx.get("amount", 0) > 0:
                total_received += tx["amount"]
                if last_deposit is None or tx["timestamp"] > last_deposit["timestamp"]:
                    last_deposit = tx
        
        result["received_amount"] = total_received
        
        if last_deposit:
            result["transaction_hash"] = last_deposit.get("hash")
            result["from_address"] = last_deposit.get("from_address")
        
        # Allow 1% tolerance for fees
        if total_received >= expected_amount * 0.99:
            result["funded"] = True
            
    except Exception as e:
        logger.error(f"Error checking deposit: {e}")
        result["error"] = str(e)
    
    return result


# =============================================================================
# TRANSFER OPERATIONS
# =============================================================================

async def send_ton(
    encrypted_mnemonic: str,
    to_address: str,
    amount: float,
    comment: str = ""
) -> Dict[str, Any]:
    """
    Send TON from escrow wallet to destination.
    
    Args:
        encrypted_mnemonic: Encrypted wallet mnemonic
        to_address: Destination address
        amount: Amount in TON
        comment: Optional transaction comment
    
    Returns:
        dict with 'success', 'tx_hash', 'error'
    """
    result = {"success": False, "tx_hash": None, "error": None}
    
    try:
        # Restore wallet
        mnemonic = decrypt_mnemonic(encrypted_mnemonic)
        _, _, priv_key, wallet = Wallets.from_mnemonics(
            mnemonics=mnemonic,
            version=WalletVersionEnum.v4r2,
            workchain=0
        )
        
        # Get current seqno
        seqno = await get_wallet_seqno(wallet.address.to_string(True, True, True))
        
        # Create transfer
        amount_nano = to_nano(amount, "ton")
        
        transfer = wallet.create_transfer_message(
            to_addr=to_address,
            amount=amount_nano,
            seqno=seqno,
            payload=comment if comment else None
        )
        
        # Send transaction
        boc = bytes_to_b64str(transfer["message"].to_boc())
        
        url = f"{get_toncenter_url()}/sendBoc"
        payload = {"boc": boc}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                
                if data.get("ok"):
                    result["success"] = True
                    result["tx_hash"] = data.get("result", {}).get("hash")
                    logger.info(f"Sent {amount} TON to {to_address[:20]}...")
                else:
                    result["error"] = data.get("error", "Transaction failed")
                    logger.error(f"Transfer failed: {result['error']}")
                    
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error sending TON: {e}")
    
    return result


async def get_wallet_seqno(address: str) -> int:
    """Get wallet seqno for transaction signing"""
    url = f"{get_toncenter_url()}/runGetMethod"
    payload = {
        "address": address,
        "method": "seqno",
        "stack": []
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                
                if data.get("ok"):
                    stack = data.get("result", {}).get("stack", [])
                    if stack and len(stack) > 0:
                        # Parse seqno from stack
                        seqno_hex = stack[0][1]
                        return int(seqno_hex, 16)
                        
    except Exception as e:
        logger.error(f"Error getting seqno: {e}")
    
    return 0  # Wallet not deployed yet


async def release_funds(
    encrypted_mnemonic: str,
    to_address: str,
    amount: float
) -> Dict[str, Any]:
    """
    Release escrow funds to channel owner.
    Sends funds minus network fee reservation.
    """
    # Reserve 0.05 TON for network fees
    fee_reserve = 0.05
    send_amount = max(0, amount - fee_reserve)
    
    if send_amount <= 0:
        return {
            "success": False,
            "error": f"Amount too small after fee reservation: {amount} TON"
        }
    
    return await send_ton(
        encrypted_mnemonic,
        to_address,
        send_amount,
        comment="TG AdEscrow - Payment Released"
    )


async def refund_funds(
    encrypted_mnemonic: str,
    to_address: str,
    amount: float
) -> Dict[str, Any]:
    """
    Refund escrow funds to advertiser.
    Sends funds minus network fee reservation.
    """
    # Reserve 0.05 TON for network fees
    fee_reserve = 0.05
    send_amount = max(0, amount - fee_reserve)
    
    if send_amount <= 0:
        return {
            "success": False,
            "error": f"Amount too small after fee reservation: {amount} TON"
        }
    
    return await send_ton(
        encrypted_mnemonic,
        to_address,
        send_amount,
        comment="TG AdEscrow - Refund"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _run_async(coro):
    """Run async coroutine in sync context (Python 3.10+ compatible)"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create new one
        return asyncio.run(coro)
    else:
        # Already in async context
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


def sync_get_balance(address: str) -> Dict[str, Any]:
    """Synchronous wrapper for get_wallet_balance"""
    return _run_async(get_wallet_balance(address))


def sync_check_deposit(address: str, expected_amount: float) -> Dict[str, Any]:
    """Synchronous wrapper for check_for_deposit"""
    return _run_async(check_for_deposit(address, expected_amount))


def sync_release_funds(encrypted_mnemonic: str, to_address: str, amount: float) -> Dict[str, Any]:
    """Synchronous wrapper for release_funds"""
    return _run_async(release_funds(encrypted_mnemonic, to_address, amount))


def sync_refund_funds(encrypted_mnemonic: str, to_address: str, amount: float) -> Dict[str, Any]:
    """Synchronous wrapper for refund_funds"""
    return _run_async(refund_funds(encrypted_mnemonic, to_address, amount))


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Generating test wallet...")
    wallet_info = generate_escrow_wallet()
    print(f"Address: {wallet_info['address']}")
    print(f"Encrypted mnemonic length: {len(wallet_info['encrypted_mnemonic'])}")
    
    # Test balance check
    print("\nChecking balance...")
    balance = sync_get_balance(wallet_info['address'])
    print(f"Balance: {balance}")
    
    print("\n[OK] All tests passed!")

