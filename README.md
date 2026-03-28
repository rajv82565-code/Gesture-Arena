# ⚔️ Gesture Arena

> A real-time AI combat game controlled entirely by your hand gestures using a webcam.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.30+-green)
![Pygame](https://img.shields.io/badge/Pygame-2.6+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎮 Gameplay

Control your fighter using nothing but hand gestures in front of your webcam:

| Gesture | Action | Effect |
|---|---|---|
| ✊ Fist (0 fingers up) | Attack | 12 HP damage |
| ✋ Palm (all fingers up) | Defend | Blocks 75% incoming damage |
| 👉 Point (index only) | Shoot | Fires a projectile (8 HP) |
| ✌️ Two fingers | Special Move | 20 HP + breaks enemy block |

Hold any gesture steady for **0.25 seconds** to trigger it.

---

## 🤖 Smart AI Enemy

The enemy uses a **3-layer AI brain** — not random, genuinely strategic:

1. **Pattern Recognition** — builds a Markov chain of your gesture history and predicts what you'll throw next
2. **Reaction Delay** — simulates real human reaction lag (fast on Hard, slow on Easy)
3. **Strategic Engine** — considers HP levels, your combo streak, and aggression weighting before acting. Displays its reasoning live on screen

---

## 🎯 Difficulty Levels

| | 🟢 Easy | 🟡 Normal | 🔴 Hard |
|---|---|---|---|
| Attack cooldown | 3.0s | 2.0s | 1.2s |
| Reaction time | 0.8s | 0.4s | 0.15s |
| Pattern trust | 30% | 65% | 92% |
| Blunder chance | 35% | 15% | 4% |
| Damage per hit | 6–12 | 10–18 | 14–24 |

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/gesture-arena.git
cd gesture-arena

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python gesture_arena.py
```

> On first run, the MediaPipe hand model (~8MB) downloads automatically. Only happens once.

---

## 📋 Requirements

- Python 3.10+
- Webcam
- Good lighting
- Hand 30–60cm from camera

---

## 🕹️ Controls

| Key | Action |
|---|---|
| Hold gesture 0.25s | Fire |
| `R` | Restart |
| `ESC` | Back to menu |

---

## 🏗️ Tech Stack

| Library | Purpose |
|---|---|
| MediaPipe | Hand landmark detection (21 keypoints) |
| OpenCV | Camera capture + frame processing |
| Pygame | Game loop, rendering, UI |
| NumPy | Frame array manipulation |

---

## 📁 Project Structure

```
gesture-arena/
├── gesture_arena.py     # Main game
├── requirements.txt     # Dependencies
├── README.md            # This file
└── .gitignore           # Excludes model file and cache
```

---

## 💡 How It Works

```
Webcam → MediaPipe (21 hand keypoints)
       → Finger state detection (tip vs pip y-comparison)
       → Gesture classification (fist/palm/point/two)
       → Hold timer (0.25s)
       → Game logic + AI Brain
       → Pygame render
```

---

## 📄 License

MIT — free to use, modify, and distribute.