"""
Runtime module for Antonio Evo.

Provides hardware-aware runtime profiles and system awareness.
"""

from .profiles import (
    RuntimeProfile,
    HardwareProfile,
    ProfileCapabilities,
    RuntimeProfileManager,
    get_profile_manager,
    PROFILE_CAPABILITIES,
)

__all__ = [
    "RuntimeProfile",
    "HardwareProfile",
    "ProfileCapabilities",
    "RuntimeProfileManager",
    "get_profile_manager",
    "PROFILE_CAPABILITIES",
]
