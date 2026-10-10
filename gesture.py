"""
╔══════════════════════════════════════════════════════╗
║           GESTURE ARENA - AI Combat Game             ║
║     Control your fighter with hand gestures!         ║
║  ✊ Fist=Attack  ✋ Palm=Defend  👉 Point=Shoot      ║
║         ✌️ Two Fingers = Special Move                ║
╚══════════════════════════════════════════════════════╝

Requirements:
    pip install opencv-python mediapipe pygame numpy

Run:
    python gesture_arena.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import urllib.request
import os
import pygame
import numpy as np
import random
import time
import sys
import math
from collections import deque

# ─────────────────────────────────────────────
#  MODEL DOWNLOAD
# ─────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8MB)... please wait.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete!")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
WIN_W, WIN_H = 960, 620
CAM_W, CAM_H = 300, 225
FPS = 60

BLACK   = (0,   0,   0  )
WHITE   = (255, 255, 255 )
RED     = (220, 50,  50  )
GREEN   = (50,  200, 80  )
BLUE    = (50,  120, 220 )
YELLOW  = (255, 220, 0   )
ORANGE  = (255, 140, 0   )
PURPLE  = (180, 60,  220 )
CYAN    = (0,   220, 220 )
GRAY    = (80,  80,  80  )
DARK    = (12,  12,  22  )
PANEL   = (22,  22,  38  )
GOLD    = (255, 200, 50  )

PLAYER_MAX_HP  = 100
ENEMY_MAX_HP   = 100
ATTACK_DAMAGE  = 12
SHOOT_DAMAGE   = 8
SPECIAL_DAMAGE = 20
DEFEND_BLOCK   = 0.75
COMBO_WINDOW   = 1.5
HOLD_THRESHOLD = 0.25

GESTURE_LABELS = {
    "fist":    ("✊ ATTACK",    RED,    ATTACK_DAMAGE),
    "palm":    ("✋ DEFEND",    BLUE,   0),
    "point":   ("👉 SHOOT",    YELLOW, SHOOT_DAMAGE),
    "two":     ("✌️ SPECIAL",  PURPLE, SPECIAL_DAMAGE),
    "unknown": ("❓ ...",       GRAY,   0),
}

# MediaPipe hand indices
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

# ─────────────────────────────────────────────
#  DIFFICULTY SETTINGS
# ─────────────────────────────────────────────
DIFFICULTY = {
    "easy": {
        "label":         "EASY",
        "color":         GREEN,
        "icon":          "🟢",
        "attack_cd":     3.0,    # seconds between AI attacks
        "ai_dmg_min":    6,
        "ai_dmg_max":    12,
        "reaction_time": 0.8,    # seconds before AI reacts to your gesture
        "pattern_weight":0.3,    # how much AI trusts pattern (0=random, 1=always pattern)
        "predict_depth": 2,      # how many recent gestures AI analyses
        "blunder_chance":0.35,   # chance AI makes a wrong decision
        "aggression":    0.35,   # probability AI chooses attack vs defend when given choice
        "description":   "Slow, makes mistakes, forgiving damage",
    },
    "normal": {
        "label":         "NORMAL",
        "color":         YELLOW,
        "icon":          "🟡",
        "attack_cd":     2.0,
        "ai_dmg_min":    10,
        "ai_dmg_max":    18,
        "reaction_time": 0.4,
        "pattern_weight":0.65,
        "predict_depth": 4,
        "blunder_chance":0.15,
        "aggression":    0.55,
        "description":   "Learns your habits, balanced difficulty",
    },
    "hard": {
        "label":         "HARD",
        "color":         RED,
        "icon":          "🔴",
        "attack_cd":     1.2,
        "ai_dmg_min":    14,
        "ai_dmg_max":    24,
        "reaction_time": 0.15,
        "pattern_weight":0.92,
        "predict_depth": 8,
        "blunder_chance":0.04,
        "aggression":    0.75,
        "description":   "Fast, predicts your moves, punishes patterns",
    },
}


# ─────────────────────────────────────────────
#  GESTURE DETECTOR
# ─────────────────────────────────────────────
class GestureDetector:
    def __init__(self, num_hands=2):
        ensure_model()
        opts = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.5,
        )
        self.detector = HandLandmarker.create_from_options(opts)
        self.current_gesture = "unknown"
        self.current_gesture_p2 = "unknown"

    def _fingers_up(self, lm):
        f = [lm[TIPS[0]].x < lm[PIPS[0]].x]
        for t, p in zip(TIPS[1:], PIPS[1:]):
            f.append(lm[t].y < lm[p].y)
        return f

    def _classify(self, lm):
        f = self._fingers_up(lm)
        c = sum(f[1:])
        if c == 0:                                              return "fist"
        if c == 4:                                              return "palm"
        if f[1] and not f[2] and not f[3] and not f[4]:        return "point"
        if f[1] and f[2] and not f[3] and not f[4]:            return "two"
        return "unknown"

    def _draw_hand(self, frame, lm, line_col=(255,220,0), dot_col=(0,255,120)):
        h, w = frame.shape[:2]
        pts = [(int(l.x*w), int(l.y*h)) for l in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], line_col, 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, dot_col, -1)

    def detect(self, frame):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = self.detector.detect(mp_img)
        g = "unknown"
        if res.hand_landmarks:
            lm = res.hand_landmarks[0]
            self._draw_hand(frame, lm)
            g = self._classify(lm)
        self.current_gesture = g
        return frame, g

    def detect_pvp(self, frame):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = self.detector.detect(mp_img)
        g1, g2 = "unknown", "unknown"
        h, w = frame.shape[:2]

        # Draw central divider line on webcam frame
        cv2.line(frame, (w // 2, 0), (w // 2, h), (90, 90, 130), 2)
        cv2.putText(frame, "P1 (LEFT)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 200, 80), 1)
        cv2.putText(frame, "P2 (RIGHT)", (w // 2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 50, 50), 1)

        if res.hand_landmarks:
            for lm in res.hand_landmarks:
                avg_x = sum(l.x for l in lm) / len(lm)
                # Video is mirrored with cv2.flip(frame, 1) before detect_pvp
                if avg_x < 0.5:
                    self._draw_hand(frame, lm, line_col=(50, 200, 80), dot_col=(0, 255, 120))
                    g1 = self._classify(lm)
                else:
                    self._draw_hand(frame, lm, line_col=(220, 50, 50), dot_col=(255, 140, 0))
                    g2 = self._classify(lm)

        self.current_gesture = g1
        self.current_gesture_p2 = g2
        return frame, g1, g2


# ─────────────────────────────────────────────
#  SMART AI ENEMY BRAIN
# ─────────────────────────────────────────────
class AIBrain:
    """
    Realistic fight AI with three layers:
      1. Pattern recognition  — tracks player's gesture history (Markov-style)
      2. Reaction system      — waits a reaction_time before countering a seen gesture
      3. Strategic decisions  — weighs attack/defend/counter based on HP, momentum, and aggression
    """
    def __init__(self, diff: dict):
        self.d = diff

        # Gesture frequency table
        self.freq   = {"fist":0, "palm":0, "point":0, "two":0}

        # Markov chain: given last gesture, what did player do next?
        # transitions[prev][next] = count
        self.transitions = {g: {"fist":0,"palm":0,"point":0,"two":0}
                            for g in ["fist","palm","point","two"]}

        # Recent gesture window for sequence analysis
        self.history = deque(maxlen=self.d["predict_depth"] * 2)

        # Pending reaction: (gesture_seen, time_seen)
        self._pending_reaction  = None
        self._last_attack_time  = time.time()

        # Internal state
        self.action       = "idle"
        self.action_label = ""
        self.is_defending = False
        self.action_timer = 0.0

        # Momentum: if player is hitting hard, AI gets more defensive
        self.player_combo_seen = 0
        self.hp_ratio          = 1.0  # updated externally
        self.player_hp_ratio   = 1.0

        self.thinking_text = "Watching..."

    # ── called every frame with the current player gesture ──
    def observe(self, gesture: str, player_combo: int):
        if gesture == "unknown":
            return
        self.player_combo_seen = player_combo

        # Update frequency
        if gesture in self.freq:
            self.freq[gesture] += 1

        # Update Markov transitions
        if len(self.history) > 0:
            prev = self.history[-1]
            if prev in self.transitions and gesture in self.transitions[prev]:
                self.transitions[prev][gesture] += 1

        self.history.append(gesture)

        # Queue a reaction with reaction_time delay (simulates human reaction lag)
        self._pending_reaction = (gesture, time.time())

    # ── predict what player will do next based on Markov chain ──
    def _predict_next(self):
        if not self.history:
            return None
        last = self.history[-1]
        trans = self.transitions.get(last, {})
        total = sum(trans.values())
        if total < 3:
            # Not enough data — fall back to most frequent
            best = max(self.freq, key=self.freq.get)
            return best if self.freq[best] > 0 else None
        # Pick most likely next gesture
        return max(trans, key=trans.get)

    # ── decide the best counter to a given gesture ──
    def _best_counter(self, gesture: str):
        """
        Game logic counters:
          fist  → defend (block) OR special (breaks guard)
          palm  → attack (they're defending, punish it)
          point → defend (block projectile)
          two   → attack fast (special is slow wind-up)
        Returns ("attack"|"defend", reasoning_string)
        """
        counters = {
            "fist":  ("defend",  "📖 You love punching — I'll block it"),
            "palm":  ("attack",  "📖 You're hiding — I'll punish your guard"),
            "point": ("defend",  "📖 Incoming shot — raising shield"),
            "two":   ("attack",  "📖 Special move incoming — I'll rush you"),
        }
        return counters.get(gesture, ("attack", "📖 Guessing..."))

    # ── main decision function, called every frame ──
    def decide(self, enemy_hp: float, player_hp: float) -> str:
        """Returns 'attack', 'defend', or 'idle'."""
        self.hp_ratio       = enemy_hp / ENEMY_MAX_HP
        self.player_hp_ratio = player_hp / PLAYER_MAX_HP
        now = time.time()

        # Cool-down between actions
        if now - self._last_attack_time < self.d["attack_cd"]:
            return "idle"

        # ── Decide based on pending reaction (what player just did) ──
        chosen_action  = None
        reasoning      = ""

        if self._pending_reaction is not None:
            seen_gesture, seen_time = self._pending_reaction
            if now - seen_time >= self.d["reaction_time"]:
                # React to what we saw
                self._pending_reaction = None

                # Should we use pattern prediction or react to current gesture?
                use_pattern = random.random() < self.d["pattern_weight"]

                if use_pattern:
                    predicted = self._predict_next()
                    if predicted:
                        chosen_action, reasoning = self._best_counter(predicted)
                        reasoning = f"🧠 Predicting you'll use {predicted}..."
                    else:
                        chosen_action, reasoning = self._best_counter(seen_gesture)
                else:
                    chosen_action, reasoning = self._best_counter(seen_gesture)

        # ── If no reaction pending, AI acts on its own based on strategy ──
        if chosen_action is None:
            # Desperate mode: low HP → more aggressive
            if self.hp_ratio < 0.25:
                chosen_action = "attack"
                reasoning     = "😤 Low HP — going all out!"
            # Punish mode: player low HP → keep attacking
            elif self.player_hp_ratio < 0.3:
                chosen_action = "attack"
                reasoning     = "😈 You're almost dead — finish it!"
            # Player on a combo streak → defend
            elif self.player_combo_seen >= 3:
                chosen_action = "defend"
                reasoning     = "🛡 You're on a streak — blocking!"
            else:
                # Default strategic choice weighted by aggression setting
                chosen_action = "attack" if random.random() < self.d["aggression"] else "defend"
                reasoning     = "⚔ Calculated attack" if chosen_action=="attack" else "🛡 Strategic guard"

        # ── Blunder: occasionally make the wrong call (only on easy/normal) ──
        if random.random() < self.d["blunder_chance"]:
            chosen_action = "defend" if chosen_action == "attack" else "attack"
            reasoning     = "😅 Misread that one..."

        # Commit to the action
        self._last_attack_time = now
        self.action       = chosen_action
        self.is_defending = (chosen_action == "defend")
        self.action_timer = 0.7
        self.thinking_text = reasoning
        return chosen_action

    def update(self, dt: float):
        self.action_timer = max(0, self.action_timer - dt)
        if self.action_timer <= 0:
            self.action       = "idle"
            self.is_defending = False


# ─────────────────────────────────────────────
#  GAME OBJECTS
# ─────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color):
        self.x,self.y = float(x),float(y)
        self.vx = random.uniform(-160,160)
        self.vy = random.uniform(-210,-40)
        self.color = color
        self.life = random.uniform(0.4,0.85)
        self.max_life = self.life
        self.size = random.randint(4,10)

    def update(self,dt):
        self.x+=self.vx*dt; self.y+=self.vy*dt; self.vy+=310*dt; self.life-=dt
        return self.life>0

    def draw(self,surf):
        a = self.life/self.max_life
        c = (int(self.color[0]*a), int(self.color[1]*a), int(self.color[2]*a))
        pygame.draw.circle(surf, c, (int(self.x),int(self.y)), max(1,int(self.size*a)))


def spawn_particles(lst, x, y, color, n=18):
    for _ in range(n):
        lst.append(Particle(x, y, color))


class Projectile:
    def __init__(self, x, y, direction=1, color=YELLOW):
        self.x, self.y = float(x), float(y)
        self.direction = direction
        self.vx = 520 * direction
        self.color = color
        self.active = True

    def update(self, dt):
        self.x += self.vx * dt
        if self.direction > 0 and self.x > WIN_W:
            self.active = False
        elif self.direction < 0 and self.x < 0:
            self.active = False

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), 9)
        pygame.draw.circle(surf, WHITE, (int(self.x), int(self.y)), 4)
        # trail
        trail_col = (180, 160, 0) if self.color == YELLOW else (210, 60, 60)
        for i in range(1, 5):
            tx = int(self.x - self.direction * i * 14)
            pygame.draw.circle(surf, trail_col, (tx, int(self.y)), max(1, 9 - i * 2))


# ─────────────────────────────────────────────
#  DRAWING UTILITIES
# ─────────────────────────────────────────────
def draw_health_bar(surf, x, y, w, h, cur, mx, color, label, font):
    ratio = max(0, cur/mx)
    pygame.draw.rect(surf, (40,40,40), (x,y,w,h), border_radius=7)
    if ratio > 0:
        c = color if ratio > 0.35 else RED
        pygame.draw.rect(surf, c, (x,y,int(w*ratio),h), border_radius=7)
        # shine
        pygame.draw.rect(surf, (255,255,255,40), (x,y,int(w*ratio),h//3), border_radius=7)
    pygame.draw.rect(surf, WHITE, (x,y,w,h), 2, border_radius=7)
    surf.blit(font.render(f"{label}  {int(cur)}/{mx}", True, WHITE), (x, y-20))


def draw_fighter(surf, x, y, color, defending, attacking, facing=1):
    shadow_col = (25,25,25)
    pygame.draw.ellipse(surf, shadow_col, (x-32,y+62,64,18))
    lean = 18*facing if attacking else 0

    # legs
    pygame.draw.line(surf, color, (x+lean,y+32), (x-18+lean,y+68), 5)
    pygame.draw.line(surf, color, (x+lean,y+32), (x+18+lean,y+68), 5)
    # torso
    pygame.draw.line(surf, color, (x+lean,y-8), (x+lean,y+32), 7)

    if defending:
        pygame.draw.line(surf, color, (x+lean,y+2), (x-28*facing+lean,y-12), 5)
        pygame.draw.line(surf, color, (x+lean,y+12), (x-28*facing+lean,y+18), 5)
        # shield glow
        for r, a in [(24,80),(20,120),(16,180)]:
            s = pygame.Surface((r*2,r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*BLUE, a), (r,r), r)
            surf.blit(s, (x-28*facing+lean-r, y+2-r))
    elif attacking:
        pygame.draw.line(surf, color, (x+lean,y), (x+50*facing+lean,y-22), 5)
        pygame.draw.line(surf, color, (x+lean,y+12), (x-22*facing+lean,y+28), 5)
        # fist glow
        s = pygame.Surface((24,24), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, 160), (12,12), 12)
        surf.blit(s, (x+50*facing+lean-12, y-22-12))
    else:
        pygame.draw.line(surf, color, (x+lean,y), (x-22*facing+lean,y+22), 5)
        pygame.draw.line(surf, color, (x+lean,y), (x+22*facing+lean,y+22), 5)

    # head
    pygame.draw.circle(surf, color, (x+lean,y-27), 23)
    pygame.draw.circle(surf, DARK,  (x+lean,y-27), 21)
    ex = 8*facing
    pygame.draw.circle(surf, color, (x+ex+lean, y-30), 5)
    pygame.draw.circle(surf, color, (x+ex+13*facing+lean, y-30), 5)
    # angry eyebrows on hard/attack
    if attacking:
        pygame.draw.line(surf, color, (x+ex+lean-4,y-38), (x+ex+lean+5,y-35), 2)


def draw_arena_bg(surf, tick):
    surf.fill(DARK)
    # animated grid
    for gx in range(0, WIN_W, 55):
        a = int(35 + 18*math.sin(tick*0.018 + gx*0.04))
        pygame.draw.line(surf, (a,a,a+18), (gx, WIN_H//2+90), (gx, WIN_H))
    for gy in range(WIN_H//2+90, WIN_H, 28):
        a = int(35 + 18*math.sin(tick*0.018 + gy*0.04))
        pygame.draw.line(surf, (a,a,a+18), (0,gy), (WIN_W,gy))
    # ground
    pygame.draw.line(surf, (55,55,100), (0,WIN_H//2+90), (WIN_W,WIN_H//2+90), 2)
    # torchlight blobs
    for tx, col in [(120,(80,30,0)), (WIN_W-120,(30,30,80))]:
        for r, al in [(60,20),(40,30),(20,50)]:
            s = pygame.Surface((r*2,r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*col,al), (r,r), r)
            surf.blit(s, (tx-r, WIN_H//2+60-r))


def draw_panel(surf, x, y, w, h, bg=PANEL, border=(60,60,110), radius=12):
    pygame.draw.rect(surf, bg, (x,y,w,h), border_radius=radius)
    pygame.draw.rect(surf, border, (x,y,w,h), 2, border_radius=radius)


# ─────────────────────────────────────────────
#  DIFFICULTY SELECTION SCREEN
# ─────────────────────────────────────────────
def difficulty_screen(screen, clock):
    font_title = pygame.font.SysFont("consolas", 52, bold=True)
    font_big   = pygame.font.SysFont("consolas", 30, bold=True)
    font_mid   = pygame.font.SysFont("consolas", 20, bold=True)
    font_small = pygame.font.SysFont("consolas", 15)
    font_tiny  = pygame.font.SysFont("consolas", 13)

    options    = ["easy", "normal", "hard"]
    selected   = 1   # default normal
    tick       = 0

    card_w, card_h = 230, 320
    card_gap       = 28
    total_w        = 3*card_w + 2*card_gap
    start_x        = WIN_W//2 - total_w//2

    while True:
        dt = clock.tick(60)/1000
        tick += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    selected = (selected-1) % 3
                if event.key == pygame.K_RIGHT:
                    selected = (selected+1) % 3
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return options[selected]
                if event.key in (pygame.K_F11, pygame.K_f):
                    pygame.display.toggle_fullscreen()
                if event.key == pygame.K_ESCAPE:
                    return "back"
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for i in range(3):
                    cx = start_x + i*(card_w+card_gap)
                    cy = WIN_H//2 - card_h//2 + 10
                    if cx <= mx <= cx+card_w and cy <= my <= cy+card_h:
                        if selected == i:
                            return options[i]
                        selected = i
            if event.type == pygame.MOUSEMOTION:
                mx, my = pygame.mouse.get_pos()
                for i in range(3):
                    cx = start_x + i*(card_w+card_gap)
                    cy = WIN_H//2 - card_h//2 + 10
                    if cx <= mx <= cx+card_w and cy <= my <= cy+card_h:
                        selected = i

        # ── Draw ──
        draw_arena_bg(screen, tick)

        # Title
        title  = font_title.render("⚔  GESTURE ARENA  ⚔", True, GOLD)
        sub    = font_mid.render("Choose your difficulty", True, (160,160,200))
        screen.blit(title, (WIN_W//2-title.get_width()//2, 40))
        screen.blit(sub,   (WIN_W//2-sub.get_width()//2,   105))

        # Cards
        for i, key in enumerate(options):
            d   = DIFFICULTY[key]
            cx  = start_x + i*(card_w+card_gap)
            cy  = WIN_H//2 - card_h//2 + 10
            sel = (i == selected)

            # Card bg + glow
            if sel:
                for gw in range(16, 0, -4):
                    s = pygame.Surface((card_w+gw*2, card_h+gw*2), pygame.SRCALPHA)
                    ac = int(30 + 20*math.sin(tick*0.08))
                    pygame.draw.rect(s, (*d["color"], ac), (0,0,card_w+gw*2,card_h+gw*2), border_radius=18)
                    screen.blit(s, (cx-gw, cy-gw))
                bg = (35,35,55)
            else:
                bg = (18,18,30)

            pygame.draw.rect(screen, bg, (cx,cy,card_w,card_h), border_radius=16)
            pygame.draw.rect(screen, d["color"] if sel else (50,50,80),
                             (cx,cy,card_w,card_h), 3 if sel else 1, border_radius=16)

            # Icon
            icon_t = font_title.render(d["icon"], True, d["color"])
            screen.blit(icon_t, (cx+card_w//2-icon_t.get_width()//2, cy+18))

            # Label
            lbl = font_big.render(d["label"], True, d["color"] if sel else (180,180,180))
            screen.blit(lbl, (cx+card_w//2-lbl.get_width()//2, cy+90))

            # Stats
            stats = [
                ("Attack CD",   f"{d['attack_cd']}s"),
                ("Reaction",    f"{d['reaction_time']}s"),
                ("Aggression",  f"{int(d['aggression']*100)}%"),
                ("AI Accuracy", f"{int((1-d['blunder_chance'])*100)}%"),
            ]
            for j,(k,v) in enumerate(stats):
                col_k = (140,140,160) if not sel else (180,180,200)
                col_v = WHITE if sel else (130,130,130)
                kt = font_small.render(k, True, col_k)
                vt = font_small.render(v, True, col_v)
                row_y = cy+130+j*26
                screen.blit(kt, (cx+16, row_y))
                screen.blit(vt, (cx+card_w-vt.get_width()-16, row_y))
                pygame.draw.line(screen, (45,45,65), (cx+14,row_y+18), (cx+card_w-14,row_y+18))

            # Description
            desc  = font_tiny.render(d["description"], True, (120,120,140) if not sel else (180,180,200))
            # word-wrap roughly
            words = d["description"].split()
            lines2 = []
            cur = ""
            for w in words:
                test = cur+" "+w if cur else w
                if font_tiny.size(test)[0] < card_w-24:
                    cur = test
                else:
                    lines2.append(cur); cur=w
            if cur: lines2.append(cur)
            for li,ln in enumerate(lines2):
                lt = font_tiny.render(ln, True, (120,120,140) if not sel else (180,180,200))
                screen.blit(lt, (cx+12, cy+242+li*16))

            # SELECTED badge
            if sel:
                badge = font_small.render("▶ SELECTED ◀", True, d["color"])
                screen.blit(badge, (cx+card_w//2-badge.get_width()//2, cy+card_h-26))

        # Controls hint
        hint = font_small.render("← → to browse   ENTER to start   ESC to mode menu   F11 to toggle fullscreen", True, (80,80,110))
        screen.blit(hint, (WIN_W//2-hint.get_width()//2, WIN_H-35))

        pygame.display.flip()


# ─────────────────────────────────────────────
#  MAIN GAME
# ─────────────────────────────────────────────
def game(screen, clock, difficulty_key: str):
    diff = DIFFICULTY[difficulty_key]

    font_big   = pygame.font.SysFont("consolas", 42, bold=True)
    font_mid   = pygame.font.SysFont("consolas", 26, bold=True)
    font_small = pygame.font.SysFont("consolas", 18)
    font_tiny  = pygame.font.SysFont("consolas", 13)

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("Camera not found."); pygame.quit(); sys.exit(1)

    detector  = GestureDetector()
    ai        = AIBrain(diff)

    player_hp           = float(PLAYER_MAX_HP)
    enemy_hp            = float(ENEMY_MAX_HP)
    particles, projs    = [], []
    last_gesture        = "unknown"
    gesture_hold_start  = time.time()
    gesture_triggered   = False
    combo_count         = 0
    combo_timer         = 0.0
    last_gesture_time   = time.time()
    damage_flash        = 0.0
    enemy_flash         = 0.0
    player_is_attacking = False
    player_atk_timer    = 0.0
    message             = ""
    msg_timer           = 0.0
    tick                = 0
    game_over           = False
    winner              = ""
    cam_surf            = None
    diff_col            = diff["color"]

    def show_msg(msg, dur=1.1):
        nonlocal message, msg_timer
        message, msg_timer = msg, dur

    def player_attack(gesture):
        nonlocal combo_count, combo_timer, last_gesture_time
        nonlocal player_is_attacking, player_atk_timer, enemy_flash, enemy_hp

        _, color, dmg = GESTURE_LABELS.get(gesture, ("", GRAY, 0))
        player_is_attacking = True
        player_atk_timer    = 0.5

        if gesture == "palm":
            show_msg("🛡️  DEFENDING!", 0.8); return

        now = time.time()
        combo_count = (combo_count+1
                       if now-last_gesture_time < COMBO_WINDOW and combo_count>0
                       else 1)
        combo_timer       = COMBO_WINDOW
        last_gesture_time = now

        actual = dmg * (1 + (combo_count-1)*0.3)

        if ai.is_defending and gesture != "two":
            actual *= 0.15
            show_msg(f"🛡️ BLOCKED! -{int(actual)} HP", 0.9)
        else:
            if gesture == "point":
                projs.append(Projectile(255, WIN_H//2+42))
                show_msg(f"💥 SHOOT! -{int(actual)} HP", 0.9)
            elif gesture == "two":
                show_msg(f"⚡ SPECIAL x{combo_count}! -{int(actual)} HP", 1.1)
                spawn_particles(particles, 650, WIN_H//2+42, PURPLE, 35)
            else:
                show_msg(f"✊ HIT x{combo_count}! -{int(actual)} HP", 0.9)
            enemy_hp = max(0, enemy_hp-actual)
            enemy_flash = 0.3
            spawn_particles(particles, 650, WIN_H//2+20, color, 22)

        ai.observe(gesture, combo_count)

    # ── GAME LOOP ──
    while True:
        dt = clock.tick(FPS)/1000.0
        tick += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release(); pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_F11, pygame.K_f):
                    pygame.display.toggle_fullscreen()
                if event.key == pygame.K_ESCAPE:
                    cap.release(); return "menu"
                if event.key == pygame.K_r and game_over:
                    cap.release(); return "restart"

        if not game_over:
            # Camera
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame, gesture = detector.detect(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (CAM_W, CAM_H))
                cam_surf = pygame.surfarray.make_surface(np.transpose(rgb,(1,0,2)))
            else:
                gesture = "unknown"

            # Gesture hold logic
            if gesture != last_gesture:
                last_gesture       = gesture
                gesture_hold_start = time.time()
                gesture_triggered  = False
            elif not gesture_triggered and gesture != "unknown":
                if time.time()-gesture_hold_start >= HOLD_THRESHOLD:
                    gesture_triggered = True
                    player_attack(gesture)

            # Timers
            combo_timer = max(0, combo_timer-dt)
            if combo_timer <= 0: combo_count = 0
            player_atk_timer = max(0, player_atk_timer-dt)
            if player_atk_timer <= 0: player_is_attacking = False

            # Projectiles
            for p in projs[:]:
                p.update(dt)
                if not p.active: projs.remove(p); continue
                if p.x > 560:
                    dmg = SHOOT_DAMAGE*(0.15 if ai.is_defending else 1.0)
                    enemy_hp = max(0, enemy_hp-dmg)
                    spawn_particles(particles, p.x, p.y, YELLOW, 16)
                    p.active = False; projs.remove(p)

            # AI brain
            ai.update(dt)
            ai_action = ai.decide(enemy_hp, player_hp)
            if ai_action == "attack":
                defending = (last_gesture == "palm")
                base_dmg  = random.uniform(diff["ai_dmg_min"], diff["ai_dmg_max"])
                dmg = base_dmg * ((1-DEFEND_BLOCK) if defending else 1.0)
                # Hard mode: AI hits harder when player is on low HP
                if difficulty_key == "hard" and player_hp < 30:
                    dmg *= 1.3
                show_msg(
                    f"🛡️ BLOCKED! -{int(dmg)} HP" if defending
                    else f"😈 ENEMY HITS! -{int(dmg)} HP", 0.9)
                player_hp = max(0, player_hp-dmg)
                damage_flash = 0.35
                spawn_particles(particles, 250, WIN_H//2+20, RED, 22)

            particles[:] = [p for p in particles if p.update(dt)]
            damage_flash = max(0, damage_flash-dt)
            enemy_flash  = max(0, enemy_flash-dt)
            msg_timer    = max(0, msg_timer-dt)

            if player_hp <= 0:  game_over, winner = True, "ENEMY"
            elif enemy_hp <= 0: game_over, winner = True, "PLAYER"

        # ══════════════════════ DRAW ══════════════════════
        draw_arena_bg(screen, tick)

        p_def = (last_gesture=="palm")
        p_atk = player_is_attacking and last_gesture not in ("palm","unknown")
        draw_fighter(screen, 240, WIN_H//2+58, GREEN, p_def, p_atk,  1)
        draw_fighter(screen, 660, WIN_H//2+58, RED,
                     ai.is_defending, ai.action=="attack", -1)

        for p in projs:     p.draw(screen)
        for p in particles: p.draw(screen)

        # Flash overlays
        if damage_flash > 0:
            s = pygame.Surface((WIN_W,WIN_H), pygame.SRCALPHA)
            s.fill((220,0,0, int(damage_flash/0.35*90)))
            screen.blit(s,(0,0))
        if enemy_flash > 0:
            s = pygame.Surface((WIN_W,WIN_H), pygame.SRCALPHA)
            s.fill((255,200,0, int(enemy_flash/0.3*50)))
            screen.blit(s,(WIN_W//2,0))

        # Health bars
        draw_health_bar(screen, 30,  45, 290, 22, player_hp, PLAYER_MAX_HP, GREEN, "YOU",   font_tiny)
        draw_health_bar(screen, 630, 45, 290, 22, enemy_hp,  ENEMY_MAX_HP,  RED,   "ENEMY", font_tiny)

        # VS + difficulty badge
        vs = font_mid.render("VS", True, YELLOW)
        screen.blit(vs, (WIN_W//2-vs.get_width()//2, 44))
        diff_badge = font_tiny.render(f"{diff['icon']} {diff['label']}", True, diff_col)
        screen.blit(diff_badge, (WIN_W//2-diff_badge.get_width()//2, 74))

        # Combo
        if combo_count > 1:
            ct = font_big.render(f"COMBO x{combo_count}!", True, ORANGE)
            screen.blit(ct, (WIN_W//2-ct.get_width()//2, 105))

        # ── Player gesture HUD (bottom-left) ──
        hx, hy = 16, WIN_H-118
        draw_panel(screen, hx, hy, 270, 98)
        lbl, col, _ = GESTURE_LABELS.get(last_gesture, ("❓ ...", GRAY, 0))
        screen.blit(font_mid.render(lbl, True, col), (hx+10, hy+10))

        # Hold progress bar
        hold_elapsed = time.time()-gesture_hold_start if last_gesture!="unknown" else 0
        hold_prog    = min(1.0, hold_elapsed/HOLD_THRESHOLD)
        bar_w = 250
        pygame.draw.rect(screen, (40,40,40), (hx+10, hy+52, bar_w, 10), border_radius=5)
        if hold_prog > 0:
            pygame.draw.rect(screen, col, (hx+10, hy+52, int(bar_w*hold_prog), 10), border_radius=5)
        screen.blit(font_tiny.render("Hold to fire →", True, (130,130,150)), (hx+10, hy+68))
        screen.blit(font_tiny.render("Controls: ESC=menu  R=restart", True, (80,80,100)), (hx+10, hy+84))

        # ── AI Brain HUD (bottom-right) ──
        ax, ay = WIN_W-290, WIN_H-168
        draw_panel(screen, ax, ay, 274, 152, border=(100,40,40))

        screen.blit(font_small.render(f"🤖 AI BRAIN [{diff['label']}]", True, diff_col), (ax+10, ay+8))

        # State
        ac_col = RED if ai.action=="attack" else BLUE if ai.action=="defend" else GRAY
        screen.blit(font_small.render(f"State: {ai.action.upper()}", True, ac_col), (ax+10, ay+32))

        # Thinking text (reasoning)
        think_words = ai.thinking_text.split()
        think_lines, cur2 = [], ""
        for w in think_words:
            test = cur2+" "+w if cur2 else w
            if font_tiny.size(test)[0] < 250: cur2=test
            else: think_lines.append(cur2); cur2=w
        if cur2: think_lines.append(cur2)
        for li,ln in enumerate(think_lines[:2]):
            screen.blit(font_tiny.render(ln, True, (170,170,200)), (ax+10, ay+56+li*15))

        # Pattern memory bars
        screen.blit(font_tiny.render("Pattern memory:", True, (120,120,150)), (ax+10, ay+92))
        bar_colors = {"fist":RED,"palm":BLUE,"point":YELLOW,"two":PURPLE}
        total_obs  = max(1, sum(ai.freq.values()))
        for bi, (gk, cnt) in enumerate(sorted(ai.freq.items(), key=lambda x:-x[1])):
            bx = ax+10+bi*64
            bh = int(30 * cnt/total_obs)
            pygame.draw.rect(screen, (40,40,40), (bx,ay+108,52,30), border_radius=4)
            if bh>0:
                pygame.draw.rect(screen, bar_colors.get(gk,GRAY), (bx,ay+108+(30-bh),52,bh), border_radius=4)
            gl = font_tiny.render(gk[:3], True, (130,130,150))
            screen.blit(gl,(bx+(52-gl.get_width())//2, ay+140))

        # Camera feed (bottom-center)
        if cam_surf:
            cx = WIN_W//2-CAM_W//2
            cy = WIN_H-CAM_H-8
            screen.blit(cam_surf, (cx,cy))
            pygame.draw.rect(screen, CYAN, (cx,cy,CAM_W,CAM_H), 2)
            screen.blit(font_tiny.render("📷 YOUR HAND", True, CYAN), (cx+5,cy+5))

        # Message banner
        if msg_timer > 0 and message:
            a  = int(255*min(1.0, msg_timer/0.35))
            ms = font_mid.render(message, True, (a,a,a))
            screen.blit(ms, (WIN_W//2-ms.get_width()//2, 160))

        # ── Gesture legend (top-right) ──
        for li,ln in enumerate(["✊ Fist=Attack","✋ Palm=Defend","👉 Point=Shoot","✌️ Two=Special"]):
            screen.blit(font_tiny.render(ln, True, (110,110,150)), (WIN_W-170, 90+li*18))

        # ── Game Over overlay ──
        if game_over:
            ov = pygame.Surface((WIN_W,WIN_H), pygame.SRCALPHA)
            ov.fill((0,0,0,175)); screen.blit(ov,(0,0))

            txt,col = ("🏆 YOU WIN!", GOLD) if winner=="PLAYER" else ("💀 YOU LOSE!", RED)
            gt = font_big.render(txt, True, col)
            screen.blit(gt, (WIN_W//2-gt.get_width()//2, WIN_H//2-80))

            # Stats summary
            stats_txt = [
                f"Difficulty: {diff['label']}",
                f"Enemy HP remaining: {int(enemy_hp)}",
                f"Your HP remaining:  {int(player_hp)}",
            ]
            for si,st in enumerate(stats_txt):
                st_s = font_small.render(st, True, (200,200,200))
                screen.blit(st_s, (WIN_W//2-st_s.get_width()//2, WIN_H//2-20+si*28))

            r1 = font_mid.render("R = play again   ESC = menu", True, WHITE)
            screen.blit(r1, (WIN_W//2-r1.get_width()//2, WIN_H//2+80))

        pygame.display.flip()

    cap.release()
    return "menu"


# ─────────────────────────────────────────────
#  MODE SELECTION SCREEN
# ─────────────────────────────────────────────
def mode_select_screen(screen, clock):
    font_title = pygame.font.SysFont("consolas", 50, bold=True)
    font_big   = pygame.font.SysFont("consolas", 28, bold=True)
    font_mid   = pygame.font.SysFont("consolas", 20, bold=True)
    font_small = pygame.font.SysFont("consolas", 15)
    font_tiny  = pygame.font.SysFont("consolas", 13)

    modes = [
        {
            "id": "single",
            "title": "SINGLE PLAYER",
            "subtitle": "VS AI Opponent",
            "icon": "🤖",
            "color": CYAN,
            "desc": "Duel against an adaptive AI fighter that learns your patterns across 3 difficulty levels.",
            "features": ["1 Player on Camera", "Adaptive AI Learning", "Easy / Normal / Hard", "Combo Streaks"],
        },
        {
            "id": "pvp",
            "title": "2-PLAYER LOCAL PVP",
            "subtitle": "Split-Screen Dual Hand",
            "icon": "⚔️",
            "color": ORANGE,
            "desc": "Battle a friend side-by-side on 1 webcam! Left hand controls P1, Right hand controls P2.",
            "features": ["2 Players on 1 Webcam", "Real-Time Duel", "Guard Breaking Specials", "Independent Hold Gauges"],
        },
    ]
    selected = 0
    tick = 0

    card_w, card_h = 360, 340
    card_gap       = 40
    total_w        = 2 * card_w + card_gap
    start_x        = WIN_W // 2 - total_w // 2

    while True:
        dt = clock.tick(60) / 1000
        tick += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_F11, pygame.K_f):
                    pygame.display.toggle_fullscreen()
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    selected = 0
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    selected = 1
                if event.key == pygame.K_1:
                    return "single"
                if event.key == pygame.K_2:
                    return "pvp"
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return modes[selected]["id"]
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for i in range(2):
                    cx = start_x + i * (card_w + card_gap)
                    cy = WIN_H // 2 - card_h // 2 + 10
                    if cx <= mx <= cx + card_w and cy <= my <= cy + card_h:
                        if selected == i:
                            return modes[i]["id"]
                        selected = i
            if event.type == pygame.MOUSEMOTION:
                mx, my = pygame.mouse.get_pos()
                for i in range(2):
                    cx = start_x + i * (card_w + card_gap)
                    cy = WIN_H // 2 - card_h // 2 + 10
                    if cx <= mx <= cx + card_w and cy <= my <= cy + card_h:
                        selected = i

        draw_arena_bg(screen, tick)

        title = font_title.render("⚔  GESTURE ARENA  ⚔", True, GOLD)
        sub   = font_mid.render("Select Game Mode", True, (160, 160, 200))
        screen.blit(title, (WIN_W // 2 - title.get_width() // 2, 35))
        screen.blit(sub,   (WIN_W // 2 - sub.get_width() // 2, 95))

        for i, m in enumerate(modes):
            cx = start_x + i * (card_w + card_gap)
            cy = WIN_H // 2 - card_h // 2 + 10
            sel = (i == selected)

            if sel:
                for gw in range(16, 0, -4):
                    s = pygame.Surface((card_w + gw * 2, card_h + gw * 2), pygame.SRCALPHA)
                    ac = int(35 + 22 * math.sin(tick * 0.08))
                    pygame.draw.rect(s, (*m["color"], ac), (0, 0, card_w + gw * 2, card_h + gw * 2), border_radius=18)
                    screen.blit(s, (cx - gw, cy - gw))
                bg = (32, 32, 52)
            else:
                bg = (18, 18, 30)

            pygame.draw.rect(screen, bg, (cx, cy, card_w, card_h), border_radius=16)
            pygame.draw.rect(screen, m["color"] if sel else (50, 50, 80),
                             (cx, cy, card_w, card_h), 3 if sel else 1, border_radius=16)

            # Icon
            icon_t = font_title.render(m["icon"], True, m["color"])
            screen.blit(icon_t, (cx + card_w // 2 - icon_t.get_width() // 2, cy + 18))

            # Title
            mt = font_big.render(m["title"], True, m["color"] if sel else (200, 200, 200))
            st = font_small.render(m["subtitle"], True, YELLOW if sel else (140, 140, 160))
            screen.blit(mt, (cx + card_w // 2 - mt.get_width() // 2, cy + 78))
            screen.blit(st, (cx + card_w // 2 - st.get_width() // 2, cy + 112))

            # Features
            for fi, feat in enumerate(m["features"]):
                fy = cy + 145 + fi * 28
                pygame.draw.circle(screen, m["color"] if sel else (80, 80, 100), (cx + 28, fy + 8), 4)
                ft = font_small.render(feat, True, WHITE if sel else (160, 160, 180))
                screen.blit(ft, (cx + 42, fy))

            # Description
            desc_words = m["desc"].split()
            lines = []
            cur = ""
            for w in desc_words:
                test = cur + " " + w if cur else w
                if font_tiny.size(test)[0] < card_w - 30:
                    cur = test
                else:
                    lines.append(cur)
                    cur = w
            if cur: lines.append(cur)
            for li, ln in enumerate(lines[:2]):
                lt = font_tiny.render(ln, True, (130, 130, 160) if not sel else (190, 190, 220))
                screen.blit(lt, (cx + 18, cy + 265 + li * 16))

            if sel:
                badge = font_small.render("▶ PRESS ENTER TO PLAY ◀", True, m["color"])
                screen.blit(badge, (cx + card_w // 2 - badge.get_width() // 2, cy + card_h - 26))

        # Bottom hint
        hint = font_small.render("Press 1 or 2  |  ← / → Arrow Keys  |  F11 Fullscreen  |  ESC Quit", True, (130, 130, 160))
        screen.blit(hint, (WIN_W // 2 - hint.get_width() // 2, WIN_H - 42))

        pygame.display.flip()


# ─────────────────────────────────────────────
#  2-PLAYER LOCAL PVP GAME
# ─────────────────────────────────────────────
def game_two_player(screen, clock):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = GestureDetector(num_hands=2)

    font_big   = pygame.font.SysFont("consolas", 40, bold=True)
    font_mid   = pygame.font.SysFont("consolas", 24, bold=True)
    font_small = pygame.font.SysFont("consolas", 16, bold=True)
    font_tiny  = pygame.font.SysFont("consolas", 13)

    p1_hp = PLAYER_MAX_HP
    p2_hp = PLAYER_MAX_HP

    projs = []
    particles = []

    p1_last_gesture = "unknown"
    p2_last_gesture = "unknown"
    p1_hold_start = 0.0
    p2_hold_start = 0.0
    p1_triggered = False
    p2_triggered = False

    p1_combo_count = 0
    p1_combo_timer = 0.0
    p1_last_gesture_time = 0.0

    p2_combo_count = 0
    p2_combo_timer = 0.0
    p2_last_gesture_time = 0.0

    p1_attacking = False
    p1_atk_timer = 0.0
    p2_attacking = False
    p2_atk_timer = 0.0

    p1_flash = 0.0
    p2_flash = 0.0

    message = ""
    msg_timer = 0.0
    game_over = False
    winner = ""
    cam_surf = None
    tick = 0

    def show_msg(msg, dur=1.1):
        nonlocal message, msg_timer
        message, msg_timer = msg, dur

    def p1_action(gesture):
        nonlocal p1_combo_count, p1_combo_timer, p1_last_gesture_time
        nonlocal p1_attacking, p1_atk_timer, p2_flash, p2_hp

        _, color, dmg = GESTURE_LABELS.get(gesture, ("", GRAY, 0))
        p1_attacking = True
        p1_atk_timer = 0.5

        if gesture == "palm":
            show_msg("P1 (LEFT): 🛡️ GUARD UP!", 0.7)
            return

        now = time.time()
        p1_combo_count = (p1_combo_count + 1
                          if now - p1_last_gesture_time < COMBO_WINDOW and p1_combo_count > 0
                          else 1)
        p1_combo_timer = COMBO_WINDOW
        p1_last_gesture_time = now

        actual = dmg * (1 + (p1_combo_count - 1) * 0.3)

        if gesture == "point":
            projs.append(Projectile(255, WIN_H // 2 + 42, direction=1, color=CYAN))
            show_msg(f"P1 💥 FIRED ENERGY BOLT!", 0.8)
        elif gesture == "two":
            # Guard-breaker! Bypasses palm block
            p2_hp = max(0, p2_hp - actual)
            p2_flash = 0.35
            spawn_particles(particles, 660, WIN_H // 2 + 30, PURPLE, 35)
            show_msg(f"⚡ P1 SPECIAL GUARD-BREAK! -{int(actual)} HP", 1.1)
        elif gesture == "fist":
            # Melee punch
            if p2_last_gesture == "palm":
                actual *= (1.0 - DEFEND_BLOCK)
                show_msg(f"P2 🛡️ BLOCKED P1 FIST! -{int(actual)} HP", 0.9)
            else:
                show_msg(f"P1 ✊ PUNCH x{p1_combo_count}! -{int(actual)} HP", 0.9)
            p2_hp = max(0, p2_hp - actual)
            p2_flash = 0.3
            spawn_particles(particles, 660, WIN_H // 2 + 30, GREEN, 22)

    def p2_action(gesture):
        nonlocal p2_combo_count, p2_combo_timer, p2_last_gesture_time
        nonlocal p2_attacking, p2_atk_timer, p1_flash, p1_hp

        _, color, dmg = GESTURE_LABELS.get(gesture, ("", GRAY, 0))
        p2_attacking = True
        p2_atk_timer = 0.5

        if gesture == "palm":
            show_msg("P2 (RIGHT): 🛡️ GUARD UP!", 0.7)
            return

        now = time.time()
        p2_combo_count = (p2_combo_count + 1
                          if now - p2_last_gesture_time < COMBO_WINDOW and p2_combo_count > 0
                          else 1)
        p2_combo_timer = COMBO_WINDOW
        p2_last_gesture_time = now

        actual = dmg * (1 + (p2_combo_count - 1) * 0.3)

        if gesture == "point":
            projs.append(Projectile(645, WIN_H // 2 + 42, direction=-1, color=ORANGE))
            show_msg(f"P2 💥 FIRED ENERGY BOLT!", 0.8)
        elif gesture == "two":
            # Guard-breaker! Bypasses palm block
            p1_hp = max(0, p1_hp - actual)
            p1_flash = 0.35
            spawn_particles(particles, 240, WIN_H // 2 + 30, PURPLE, 35)
            show_msg(f"⚡ P2 SPECIAL GUARD-BREAK! -{int(actual)} HP", 1.1)
        elif gesture == "fist":
            # Melee punch
            if p1_last_gesture == "palm":
                actual *= (1.0 - DEFEND_BLOCK)
                show_msg(f"P1 🛡️ BLOCKED P2 FIST! -{int(actual)} HP", 0.9)
            else:
                show_msg(f"P2 ✊ PUNCH x{p2_combo_count}! -{int(actual)} HP", 0.9)
            p1_hp = max(0, p1_hp - actual)
            p1_flash = 0.3
            spawn_particles(particles, 240, WIN_H // 2 + 30, RED, 22)

    # ── GAME LOOP ──
    while True:
        dt = clock.tick(FPS) / 1000.0
        tick += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release(); pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_F11, pygame.K_f):
                    pygame.display.toggle_fullscreen()
                if event.key == pygame.K_ESCAPE:
                    cap.release(); return "menu"
                if event.key == pygame.K_r and game_over:
                    cap.release(); return "restart"

        if not game_over:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame, g1, g2 = detector.detect_pvp(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (CAM_W, CAM_H))
                cam_surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            else:
                g1, g2 = "unknown", "unknown"

            # P1 gesture hold
            if g1 != p1_last_gesture:
                p1_last_gesture = g1
                p1_hold_start = time.time()
                p1_triggered = False
            elif not p1_triggered and g1 != "unknown":
                if time.time() - p1_hold_start >= HOLD_THRESHOLD:
                    p1_triggered = True
                    p1_action(g1)

            # P2 gesture hold
            if g2 != p2_last_gesture:
                p2_last_gesture = g2
                p2_hold_start = time.time()
                p2_triggered = False
            elif not p2_triggered and g2 != "unknown":
                if time.time() - p2_hold_start >= HOLD_THRESHOLD:
                    p2_triggered = True
                    p2_action(g2)

            # Timers
            p1_combo_timer = max(0, p1_combo_timer - dt)
            if p1_combo_timer <= 0: p1_combo_count = 0
            p1_atk_timer = max(0, p1_atk_timer - dt)
            p1_attacking = (p1_atk_timer > 0)

            p2_combo_timer = max(0, p2_combo_timer - dt)
            if p2_combo_timer <= 0: p2_combo_count = 0
            p2_atk_timer = max(0, p2_atk_timer - dt)
            p2_attacking = (p2_atk_timer > 0)

            # Projectiles
            for p in projs:
                p.update(dt)
                if p.active:
                    # P1 projectile moving right towards P2 (x=660)
                    if p.direction > 0 and p.x >= 640:
                        p.active = False
                        if p2_last_gesture == "palm":
                            actual = SHOOT_DAMAGE * (1.0 - DEFEND_BLOCK)
                            show_msg(f"P2 🛡️ BLOCKED P1 BOLT! -{int(actual)} HP", 0.9)
                        else:
                            actual = SHOOT_DAMAGE
                            show_msg(f"P1 💥 BOLT HIT P2! -{int(actual)} HP", 0.9)
                        p2_hp = max(0, p2_hp - actual)
                        p2_flash = 0.3
                        spawn_particles(particles, 660, WIN_H // 2 + 42, CYAN, 20)
                    # P2 projectile moving left towards P1 (x=240)
                    elif p.direction < 0 and p.x <= 260:
                        p.active = False
                        if p1_last_gesture == "palm":
                            actual = SHOOT_DAMAGE * (1.0 - DEFEND_BLOCK)
                            show_msg(f"P1 🛡️ BLOCKED P2 BOLT! -{int(actual)} HP", 0.9)
                        else:
                            actual = SHOOT_DAMAGE
                            show_msg(f"P2 💥 BOLT HIT P1! -{int(actual)} HP", 0.9)
                        p1_hp = max(0, p1_hp - actual)
                        p1_flash = 0.3
                        spawn_particles(particles, 240, WIN_H // 2 + 42, ORANGE, 20)

            projs[:] = [p for p in projs if p.active]
            particles[:] = [p for p in particles if p.update(dt)]
            p1_flash = max(0, p1_flash - dt)
            p2_flash = max(0, p2_flash - dt)
            msg_timer = max(0, msg_timer - dt)

            if p1_hp <= 0:
                game_over, winner = True, "PLAYER 2"
            elif p2_hp <= 0:
                game_over, winner = True, "PLAYER 1"

        # ══════════════════════ DRAW ══════════════════════
        draw_arena_bg(screen, tick)

        p1_def = (p1_last_gesture == "palm")
        p1_atk = p1_attacking and p1_last_gesture not in ("palm", "unknown")
        draw_fighter(screen, 240, WIN_H // 2 + 58, GREEN, p1_def, p1_atk, 1)

        p2_def = (p2_last_gesture == "palm")
        p2_atk = p2_attacking and p2_last_gesture not in ("palm", "unknown")
        draw_fighter(screen, 660, WIN_H // 2 + 58, RED, p2_def, p2_atk, -1)

        for p in projs:     p.draw(screen)
        for p in particles: p.draw(screen)

        # Flash overlays
        if p1_flash > 0:
            s1 = pygame.Surface((WIN_W // 2, WIN_H), pygame.SRCALPHA)
            s1.fill((220, 0, 0, int(p1_flash / 0.35 * 90)))
            screen.blit(s1, (0, 0))
        if p2_flash > 0:
            s2 = pygame.Surface((WIN_W // 2, WIN_H), pygame.SRCALPHA)
            s2.fill((220, 0, 0, int(p2_flash / 0.35 * 90)))
            screen.blit(s2, (WIN_W // 2, 0))

        # Health bars
        draw_health_bar(screen, 30, 45, 290, 22, p1_hp, PLAYER_MAX_HP, GREEN, "PLAYER 1 (LEFT)", font_tiny)
        draw_health_bar(screen, 630, 45, 290, 22, p2_hp, PLAYER_MAX_HP, RED, "PLAYER 2 (RIGHT)", font_tiny)

        # VS + mode badge
        vs = font_mid.render("VS", True, YELLOW)
        screen.blit(vs, (WIN_W // 2 - vs.get_width() // 2, 44))
        pvp_badge = font_tiny.render("⚔️ 2-PLAYER LOCAL PVP ⚔️", True, ORANGE)
        screen.blit(pvp_badge, (WIN_W // 2 - pvp_badge.get_width() // 2, 74))

        # Legend at top
        legend_txt = font_tiny.render("✊ Fist=Attack  |  ✋ Palm=Defend  |  👉 Point=Shoot  |  ✌️ Two=Special", True, (160, 160, 190))
        screen.blit(legend_txt, (WIN_W // 2 - legend_txt.get_width() // 2, 18))

        # Combo alerts
        if p1_combo_count > 1:
            c1 = font_small.render(f"P1 COMBO x{p1_combo_count}!", True, GREEN)
            screen.blit(c1, (30, 80))
        if p2_combo_count > 1:
            c2 = font_small.render(f"P2 COMBO x{p2_combo_count}!", True, RED)
            screen.blit(c2, (WIN_W - 30 - c2.get_width(), 80))

        # ── Player 1 HUD (bottom-left) ──
        hx1, hy1 = 16, WIN_H - 118
        draw_panel(screen, hx1, hy1, 270, 98, border=(50, 200, 80))
        lbl1, col1, _ = GESTURE_LABELS.get(p1_last_gesture, ("❓ ...", GRAY, 0))
        screen.blit(font_small.render(f"P1: {lbl1}", True, col1), (hx1 + 10, hy1 + 10))

        hold_el1 = time.time() - p1_hold_start if p1_last_gesture != "unknown" else 0
        prog1    = min(1.0, hold_el1 / HOLD_THRESHOLD)
        bar_w = 250
        pygame.draw.rect(screen, (40, 40, 40), (hx1 + 10, hy1 + 44, bar_w, 10), border_radius=5)
        if prog1 > 0:
            pygame.draw.rect(screen, col1, (hx1 + 10, hy1 + 44, int(bar_w * prog1), 10), border_radius=5)
        screen.blit(font_tiny.render("Hold to fire →", True, (130, 130, 160)), (hx1 + 10, hy1 + 60))
        screen.blit(font_tiny.render("Controls: Left Webcam Side", True, (100, 200, 120)), (hx1 + 10, hy1 + 78))

        # ── Player 2 HUD (bottom-right) ──
        hx2, hy2 = WIN_W - 286, WIN_H - 118
        draw_panel(screen, hx2, hy2, 270, 98, border=(220, 50, 50))
        lbl2, col2, _ = GESTURE_LABELS.get(p2_last_gesture, ("❓ ...", GRAY, 0))
        screen.blit(font_small.render(f"P2: {lbl2}", True, col2), (hx2 + 10, hy2 + 10))

        hold_el2 = time.time() - p2_hold_start if p2_last_gesture != "unknown" else 0
        prog2    = min(1.0, hold_el2 / HOLD_THRESHOLD)
        pygame.draw.rect(screen, (40, 40, 40), (hx2 + 10, hy2 + 44, bar_w, 10), border_radius=5)
        if prog2 > 0:
            pygame.draw.rect(screen, col2, (hx2 + 10, hy2 + 44, int(bar_w * prog2), 10), border_radius=5)
        screen.blit(font_tiny.render("Hold to fire →", True, (130, 130, 160)), (hx2 + 10, hy2 + 60))
        screen.blit(font_tiny.render("Controls: Right Webcam Side", True, (240, 110, 110)), (hx2 + 10, hy2 + 78))

        # Camera feed (bottom-center)
        if cam_surf:
            cx = WIN_W // 2 - CAM_W // 2
            cy = WIN_H - CAM_H - 8
            screen.blit(cam_surf, (cx, cy))
            pygame.draw.rect(screen, GOLD, (cx, cy, CAM_W, CAM_H), 2)
            screen.blit(font_tiny.render("📷 P1 (LEFT)  |  P2 (RIGHT)", True, GOLD), (cx + 5, cy + 5))

        # Message banner
        if msg_timer > 0 and message:
            a = int(255 * min(1.0, msg_timer / 0.35))
            ms = font_mid.render(message, True, (a, a, a))
            screen.blit(ms, (WIN_W // 2 - ms.get_width() // 2, 140))

        # ── Game Over overlay ──
        if game_over:
            ov = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 185)); screen.blit(ov, (0, 0))

            wcol = GREEN if winner == "PLAYER 1" else RED
            gt = font_big.render(f"🏆 {winner} WINS! 🏆", True, wcol)
            screen.blit(gt, (WIN_W // 2 - gt.get_width() // 2, WIN_H // 2 - 75))

            stats_txt = [
                f"Player 1 remaining HP: {int(p1_hp)}",
                f"Player 2 remaining HP: {int(p2_hp)}",
            ]
            for si, st in enumerate(stats_txt):
                st_s = font_small.render(st, True, (200, 200, 200))
                screen.blit(st_s, (WIN_W // 2 - st_s.get_width() // 2, WIN_H // 2 - 15 + si * 28))

            r1 = font_mid.render("R = Rematch   ESC = Mode Menu", True, WHITE)
            screen.blit(r1, (WIN_W // 2 - r1.get_width() // 2, WIN_H // 2 + 65))

        pygame.display.flip()

    cap.release()
    return "menu"


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    pygame.init()
    # Opens in Fullscreen mode by default (scaled smoothly to fit your laptop screen)
    # Press F11 (or F) anytime to toggle between Fullscreen and Windowed!
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.FULLSCREEN | pygame.SCALED)
    pygame.display.set_caption("⚔️  GESTURE ARENA  ⚔️")
    clock  = pygame.time.Clock()

    while True:
        mode = mode_select_screen(screen, clock)
        if mode == "single":
            while True:
                diff_key = difficulty_screen(screen, clock)
                if diff_key == "back":
                    break
                result = game(screen, clock, diff_key)
                if result == "menu":
                    break
                elif result == "restart":
                    continue
        elif mode == "pvp":
            while True:
                result = game_two_player(screen, clock)
                if result == "menu":
                    break
                elif result == "restart":
                    continue


if __name__ == "__main__":
    main()