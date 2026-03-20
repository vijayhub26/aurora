"""
gesture_engine.py — Fingertip Cursor Engine
============================================
The user's index-fingertip acts as a cursor on screen.
  - Point at a ZONE  → that appliance becomes active
  - Move finger UP/DOWN inside the zone → adjusts the value
  - Hold FIST for ~15 frames → toggles power for active appliance
  - All 4 fingers spread → master OFF
"""

import cv2
import math
import numpy as np
from collections import deque


# ─── LANDMARK IDs ────────────────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4
INDEX_MCP  = 5
INDEX_TIP  = 8
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20

# UI colours
C_INACTIVE = (60,  60,  60)
C_ACTIVE   = (0,  210, 255)
C_ON       = (0,  230,  80)
C_OFF      = (0,   0,  180)
C_TEXT     = (240, 240, 240)
C_CURSOR   = (0,  210, 255)


class ApplianceController:
    def __init__(self):
        self.light_on   = False
        self.brightness = 128   # 0–255
        self.fan_on     = False
        self.fan_speed  = 50    # 0–100
        self.volume     = 50    # 0–100


class Zone:
    """A rectangular control zone on screen."""
    def __init__(self, label, x, y, w, h, var_key, var_range, appliance_key):
        self.label        = label
        self.rect         = (x, y, w, h)   # (x, y, width, height) top-left
        self.var_key      = var_key         # 'brightness' | 'fan_speed' | 'volume'
        self.lo, self.hi  = var_range
        self.appliance    = appliance_key   # 'light' | 'fan' | 'speaker'
        self.active       = False

    def contains(self, px, py):
        x, y, w, h = self.rect
        return x <= px <= x + w and y <= py <= y + h

    def value_from_y(self, py):
        """Map fingertip Y → value; high on screen = high value."""
        x, y, w, h = self.rect
        # clamp y to zone
        rel = np.clip((y + h - py) / h, 0.0, 1.0)   # 0=bottom, 1=top
        return int(np.interp(rel, [0, 1], [self.lo, self.hi]))


class GestureEngine:
    def __init__(self, frame_w=640, frame_h=480):
        self.a     = ApplianceController()
        self.w     = frame_w
        self.h     = frame_h

        # ── Define the three zones (below the guide strip at top) ──
        zone_y  = 100          # top of zone band
        zone_h  = 160          # height of zone band
        pad     = 20
        zw      = (frame_w - 4 * pad) // 3

        self.zones = [
            Zone("💡 LIGHT",   pad,             zone_y, zw, zone_h,
                 'brightness', (0, 255), 'light'),
            Zone("🌀 FAN",     2*pad + zw,      zone_y, zw, zone_h,
                 'fan_speed',  (0, 100), 'fan'),
            Zone("🔊 VOLUME",  3*pad + 2*zw,    zone_y, zw, zone_h,
                 'volume',     (0, 100), 'speaker'),
        ]

        self.active_zone  = None
        self.smooth_buf   = deque(maxlen=8)

        # Fist-hold logic
        self.fist_frames  = 0
        self.FIST_HOLD    = 15

        # Cursor display
        self.cursor_px    = (-1, -1)

    # ─── GEOMETRY HELPERS ────────────────────────────────────────────────────

    def _dist(self, a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _finger_extended(self, lm, tip, mcp):
        return lm[tip].y < lm[mcp].y - 0.04

    def _is_fist(self, lm):
        pairs = [(INDEX_TIP,5),(MIDDLE_TIP,9),(RING_TIP,13),(PINKY_TIP,17)]
        return not any(self._finger_extended(lm, t, m) for t, m in pairs)

    def _is_open_palm(self, lm):
        pairs = [(INDEX_TIP,5),(MIDDLE_TIP,9),(RING_TIP,13),(PINKY_TIP,17)]
        return all(self._finger_extended(lm, t, m) for t, m in pairs)

    # ─── MAIN UPDATE ─────────────────────────────────────────────────────────

    def process(self, lm, frame_w, frame_h):
        """
        lm       : list of 21 MediaPipe landmark objects
        frame_w/h: actual frame dimensions (may differ from init)
        Returns  : (cursor_px, active_zone_label | None)
        """
        # ── Fist: toggle power on current zone's appliance ──
        if self._is_fist(lm):
            self.fist_frames += 1
            if self.fist_frames == self.FIST_HOLD and self.active_zone:
                appl = self.active_zone.appliance
                if appl == 'light':
                    self.a.light_on = not self.a.light_on
                elif appl == 'fan':
                    self.a.fan_on = not self.a.fan_on
            self.cursor_px = (-1, -1)
            return self.cursor_px, None
        else:
            self.fist_frames = 0

        # ── Open Palm: master OFF ──
        if self._is_open_palm(lm):
            self.a.light_on = False
            self.a.fan_on   = False
            self.cursor_px  = (-1, -1)
            return self.cursor_px, "ALL OFF"

        # ── Fingertip cursor (index tip) ──
        tip    = lm[INDEX_TIP]
        px     = int(tip.x * frame_w)
        py     = int(tip.y * frame_h)
        self.cursor_px = (px, py)

        # ── Zone hit test ──
        hit_zone = None
        for z in self.zones:
            z.active = False
            if z.contains(px, py):
                hit_zone = z

        label = None
        if hit_zone:
            hit_zone.active = True
            self.active_zone = hit_zone

            # Smooth the value
            raw = hit_zone.value_from_y(py)
            self.smooth_buf.append(raw)
            smoothed = int(np.mean(self.smooth_buf))

            # Write to appliance
            vk = hit_zone.var_key
            if vk == 'brightness':
                self.a.brightness = smoothed
            elif vk == 'fan_speed':
                self.a.fan_speed = smoothed
            elif vk == 'volume':
                self.a.volume = smoothed

            label = hit_zone.label
        else:
            # Outside all zones → freeze value, clear smoother
            if self.active_zone and self.active_zone.active is False:
                self.smooth_buf.clear()
            self.active_zone = None

        return self.cursor_px, label

    # ─── AR OVERLAY ──────────────────────────────────────────────────────────

    def draw_ui(self, frame):
        a = self.a

        # Guide strip at top
        cv2.rectangle(frame, (0,0), (frame.shape[1], 90), (20,20,20), -1)
        guides = [
            ("☝ POINT  → Select control", (10, 25)),
            ("↕ MOVE   → Adjust value",   (10, 50)),
            ("✊ FIST   → Toggle ON/OFF",  (10, 75)),
        ]
        for txt, pos in guides:
            cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170,170,170), 1)

        # ── Draw each zone ──
        for z in self.zones:
            x, y, w, h = z.rect
            bdr   = C_ACTIVE if z.active else C_INACTIVE
            thick = 2 if z.active else 1

            # Background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x,y), (x+w, y+h), (30,30,30), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Border
            cv2.rectangle(frame, (x,y), (x+w, y+h), bdr, thick)

            # Retrieve value
            vk    = z.var_key
            value = getattr(a, vk)       # brightness / fan_speed / volume
            hi    = z.hi
            pct   = value / hi

            # Fill bar (vertical)
            bar_top = int(y + h * (1 - pct))
            cv2.rectangle(frame, (x+4, bar_top), (x+w-4, y+h-4), bdr, -1)

            # Label
            cv2.putText(frame, z.label, (x+6, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_TEXT, 1)

            # Value text
            if vk == 'brightness':
                pwr_c = C_ON if a.light_on else C_OFF
                cv2.putText(frame, f"{'ON' if a.light_on else 'OFF'}",
                            (x+6, y+38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, pwr_c, 1)
                cv2.putText(frame, f"{int(pct*100)}%",
                            (x+6, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 2)
            elif vk == 'fan_speed':
                pwr_c = C_ON if a.fan_on else C_OFF
                cv2.putText(frame, f"{'ON' if a.fan_on else 'OFF'}",
                            (x+6, y+38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, pwr_c, 1)
                cv2.putText(frame, f"{value}%",
                            (x+6, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 2)
            elif vk == 'volume':
                cv2.putText(frame, f"{value}%",
                            (x+6, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 2)

        # ── Draw cursor ──
        cx, cy = self.cursor_px
        if cx >= 0:
            cv2.circle(frame, (cx, cy), 12, C_CURSOR, 2)
            cv2.circle(frame, (cx, cy),  3, C_CURSOR, -1)
            # Crosshair lines
            cv2.line(frame, (cx-18, cy), (cx-12, cy), C_CURSOR, 1)
            cv2.line(frame, (cx+12, cy), (cx+18, cy), C_CURSOR, 1)
            cv2.line(frame, (cx, cy-18), (cx, cy-12), C_CURSOR, 1)
            cv2.line(frame, (cx, cy+12), (cx, cy+18), C_CURSOR, 1)

        # ── Fist hold progress bar ──
        if self.fist_frames > 0:
            prog = int(self.fist_frames / self.FIST_HOLD * 200)
            cv2.rectangle(frame, (frame.shape[1]//2 - 100, 275),
                          (frame.shape[1]//2 - 100 + prog, 295), (255,100,0), -1)
            cv2.putText(frame, "HOLD TO TOGGLE...",
                        (frame.shape[1]//2 - 90, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,150,50), 1)

        return frame
