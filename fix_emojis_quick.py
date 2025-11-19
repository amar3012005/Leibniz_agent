#!/usr/bin/env python3
import re

# Read the file
with open('leibniz_agent/leibniz_pro.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace common emojis with text
replacements = {
    '✅': 'OK',
    '❌': 'ERROR',
    '⚠️': 'WARNING',
    '🎯': 'TARGET',
    '🎙️': 'MIC',
    '📝': 'NOTE',
    '🔄': 'RETRY',
    '🧹': 'CLEAN',
    '▶️': 'PLAY',
    '⏱️': 'TIMER',
    '📊': 'DATA',
    '🚀': 'START',
    '🔥': 'HOT',
    '💻': 'PC',
    '🎓': 'GRAD',
    '🗣️': 'SPEAK',
    '📄': 'DOC',
    '⚡': 'FAST',
    '⏳': 'WAIT',
    '🕐': 'TIME',
    '🔍': 'SEARCH',
    '🧠': 'BRAIN',
    '📚': 'BOOKS',
}

for emoji, text in replacements.items():
    content = content.replace(emoji, text)

# Write back
with open('leibniz_agent/leibniz_pro.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Emojis replaced in leibniz_pro.py")
