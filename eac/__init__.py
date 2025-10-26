"""
Equity-Aware Checkout (EAC) - AI Agentic Framework

A zero-touch personalization system that integrates privacy-preserving 
Social Determinants of Health (SDOH) signals to reduce cost, risk, and 
time-to-access for essential goods.
"""

__version__ = "0.1.0"
__author__ = "EAC Research Team"

from eac.agent import EACAgent
from eac.config import EACConfig

__all__ = ["EACAgent", "EACConfig"]
