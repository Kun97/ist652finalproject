#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IST 652 Final Project

@author: Kun Yang
"""

import pandas as pd

data = pd.read_csv('/Users/kun/OneDrive - Syracuse University/IST 652 Scripting for Data Analysis/Project/data.csv')

data['blueWins'] = data['blueWins'].map({1: 'Yes', 0: 'No'})

data['blueTotalGold'] = pd.cut(data['blueTotalGold'], bins = 5, labels = ['Very Few', 'Few', 'Normal', 'Many', 'Very Many'])

data['blueTotalExperience'] = pd.cut(data['blueTotalExperience'], bins = 3, labels = ['Low', 'Normal', 'High'])

data['blueTotalMinionsKilled'] = pd.cut(data['blueTotalMinionsKilled'], bins = 3, labels = ['Low', 'Normal', 'High'])

data['blueTotalJungleMinionsKilled'] = pd.cut(data['blueTotalJungleMinionsKilled'], bins = 3, labels = ['Low', 'Normal', 'High'])

data['blueCSPerMin'] = pd.cut(data['blueCSPerMin'], bins = 3, labels = ['Low', 'Normal', 'High'])

data['blueGoldPerMin'] = pd.cut(data['blueGoldPerMin'], bins = 5, labels = ['Very Few', 'Few', 'Normal', 'Many', 'Very Many'])

data['blueFirstBlood'] = data['blueFirstBlood'].map({1: 'Yes', 0: 'No'})

data.to_csv('/Users/kun/OneDrive - Syracuse University/IST 652 Scripting for Data Analysis/Project/data1.csv', )
