# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:53:22 2024

@author: arnav
"""

from pydantic import BaseModel

class Crop_Recommender(BaseModel):
    N : int
    P : int
    K : int
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    