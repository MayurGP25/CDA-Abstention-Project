"""Logit-level measurement of abstention survival under constrained decoding.

Reframes grammar-constrained jailbreaks (EnumAttack/DictAttack) as a controlled
instrument: mask the refusal channel, then read the *pre-mask* next-token
distribution to ask whether the abstention signal is latent (present but masked)
or surface (gone once its opening tokens are unavailable).
"""

__version__ = "0.1.0"
