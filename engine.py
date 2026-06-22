import cv2
import numpy as np
import time
import collections

class BackgroundModel:
    def __init__(self, h, w, n_frames=90):
        self.buf   = collections.deque(maxlen=n_frames)
        self.bg    = None
        self.ready = False
        self.h, self.w = h, w
        self._tick = 0

    def update(self, frame_f32):
        self.buf.append(frame_f32)
        self._tick += 1
        if len(self.buf) >= 15 and self._tick % 6 == 0:
            self.bg    = np.mean(self.buf, axis=0).astype(np.float32)
            self.ready = True

    def get(self):
        return self.bg if self.ready else None


SEG_W, SEG_H = 320, 180

class SegmentationEngine:
    def __init__(self):
        import mediapipe as mp
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self._prev_mask = None
        self._frame_idx = 0

    def get_mask(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._frame_idx += 1
        if self._frame_idx % 2 == 0 and self._prev_mask is not None:
            return self._prev_mask
        small = cv2.resize(frame_bgr, (SEG_W, SEG_H), interpolation=cv2.INTER_AREA)
        res   = self.seg.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if res.segmentation_mask is None:
            return self._prev_mask if self._prev_mask is not None else \
                   np.zeros((h, w), dtype=np.float32)
        mask = cv2.resize(res.segmentation_mask, (w, h),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if self._prev_mask is not None:
            mask = 0.6 * mask + 0.4 * self._prev_mask
        _, hard = cv2.threshold(mask, 0.25, 1.0, cv2.THRESH_BINARY)
        hard8 = (hard * 255).astype(np.uint8)
        kernel = np.ones((15, 15), np.uint8)
        hard8 = cv2.morphologyEx(hard8, cv2.MORPH_CLOSE, kernel)
        hard8 = cv2.dilate(hard8, kernel, iterations=1)
        hard = hard8.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(hard, (9, 9), 0)
        mask = np.clip(mask * 1.3, 0, 1).astype(np.float32)
        self._prev_mask = mask
        return mask


HAND_W = 480
PINCH_RATIO_THRESHOLD = 0.45

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

class HandTracker:
    def __init__(self):
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )

    def process(self, frame_bgr):
        h, w  = frame_bgr.shape[:2]
        sw    = min(w, HAND_W)
        sh    = int(h * sw / w)
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        return self.hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    def get_info(self, results, w, h):
        out = {"hand_count": 0, "tips": None, "fingers_touching": False, "all_points": []}
        if not results or not results.multi_hand_landmarks:
            return out

        out["hand_count"] = len(results.multi_hand_landmarks)
        index_tips = []
        any_pinch  = False

        for lms in results.multi_hand_landmarks:
            lm = lms.landmark
            thumb_tip = lm[4]
            index_tip = lm[8]
            wrist     = lm[0]
            mid_mcp   = lm[9]

            index_tips.append((int(index_tip.x * w), int(index_tip.y * h)))
            out["all_points"].append([(int(p.x * w), int(p.y * h)) for p in lm])

            palm_size  = ((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)**0.5
            pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

            if palm_size > 1e-6:
                ratio = pinch_dist / palm_size
                if ratio < PINCH_RATIO_THRESHOLD:
                    any_pinch = True

        out["fingers_touching"] = any_pinch

        if len(index_tips) >= 2:
            out["tips"] = (index_tips[0], index_tips[1])

        return out


def draw_hand_mesh(frame, all_points):
    if not all_points:
        return frame
    for pts in all_points:
        if len(pts) < 21:
            continue
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = 6 if i in (4, 8) else 3
            cv2.circle(frame, p, r, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(frame, p, r, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class PortalBox:
    def __init__(self):
        self.tl           = None
        self.br           = None
        self.active       = False
        self.invisible    = False
        self._alpha       = 0.0
        self._scan_offset = 0
        self._touch_cd    = 0

    def update(self, info: dict):
        tips             = info["tips"]
        fingers_touching = info["fingers_touching"]
        hand_count       = info["hand_count"]

        if tips is not None:
            p1, p2 = tips
            self.tl = (min(p1[0], p2[0]), min(p1[1], p2[1]))
            self.br = (max(p1[0], p2[0]), max(p1[1], p2[1]))
            w_box = self.br[0] - self.tl[0]
            h_box = self.br[1] - self.tl[1]
            self.active = w_box > 80 and h_box > 80
        else:
            self.active = False

        if self._touch_cd > 0:
            self._touch_cd -= 1

        if fingers_touching and hand_count >= 1 and self._touch_cd == 0:
            self.invisible = not self.invisible
            self._touch_cd = 20

    def update_alpha(self):
        target = 1.0 if self.invisible else 0.0
        speed  = 0.10
        if self._alpha < target:
            self._alpha = min(target, self._alpha + speed)
        else:
            self._alpha = max(target, self._alpha - speed)

    def render(self, frame, seg_mask, bg, all_points=None):
        h, w  = frame.shape[:2]
        alpha = self._alpha

        if alpha > 0.01 and bg is not None:
            roi_f    = frame.astype(np.float32)
            roi_bg   = bg
            eff_mask = seg_mask if alpha < 0.97 else np.minimum(seg_mask * 1.6, 1.0)
            roi_mask = eff_mask[:, :, np.newaxis]
            blend    = roi_f * (1 - roi_mask * alpha) + roi_bg * (roi_mask * alpha)
            np.clip(blend, 0, 255, out=blend)
            frame[:, :] = blend.astype(np.uint8)
            if alpha > 0.05:
                self._draw_scanlines(frame, seg_mask, 0, 0, w, h, alpha)

        if all_points:
            draw_hand_mesh(frame, all_points)

        if self.active and self.tl and self.br:
            a   = max(0.4, 1.0 - self._alpha * 0.5)
            x1, y1 = self.tl; x2, y2 = self.br

            for thick, blend_str, color in [
                (11, 0.12, (0, int(180*a), int(255*a))),
                (6,  0.30, (0, int(210*a), int(255*a))),
                (2,  0.70, (0, int(230*a), int(255*a))),
                (1,  1.00, (255, 255, 255)),
            ]:
                ov = frame.copy()
                cv2.rectangle(ov, (x1,y1), (x2,y2), color, thick)
                cv2.addWeighted(ov, blend_str * a, frame, 1 - blend_str * a, 0, frame)

            l    = max(25, min(50, (x2-x1)//5, (y2-y1)//5))
            ccol = (0, int(240*a), int(255*a))
            for (cx, cy), (ddx1, ddy1), (ddx2, ddy2) in [
                ((x1, y1), ( l,  0), ( 0,  l)),
                ((x2, y1), (-l,  0), ( 0,  l)),
                ((x1, y2), ( l,  0), ( 0, -l)),
                ((x2, y2), (-l,  0), ( 0, -l)),
            ]:
                cv2.line(frame, (cx, cy), (cx+ddx1, cy+ddy1), ccol, 3, cv2.LINE_AA)
                cv2.line(frame, (cx, cy), (cx+ddx2, cy+ddy2), ccol, 3, cv2.LINE_AA)

            msg = "PINCH THUMB + INDEX  ->  VANISH" if not self.invisible else "PINCH AGAIN  ->  REAPPEAR"
            tw  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
            mx  = (x1 + x2) // 2 - tw // 2
            my  = (y1 + y2) // 2
            cv2.putText(frame, msg, (mx+1, my+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, msg, (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (0, int(230*a), int(255*a)), 1, cv2.LINE_AA)

        self._scan_offset = (self._scan_offset + 3) % max(1, frame.shape[0])
        return frame

    def _draw_scanlines(self, frame, seg_mask, x1, y1, x2, y2, alpha):
        scan_col = np.array([0, 160, 220], dtype=np.float32)
        roi_h = y2 - y1
        if roi_h <= 0:
            return
        rows = (np.arange(0, roi_h, 6) + self._scan_offset) % roi_h + y1
        rows = rows[rows < y2].astype(int)
        if rows.size == 0:
            return
        sm    = seg_mask[rows, x1:x2]
        strip = frame[rows, x1:x2].astype(np.float32)
        t     = 0.10 * alpha
        strip = strip*(1 - t*sm[:,:,None]) + scan_col*t*sm[:,:,None]
        frame[rows, x1:x2] = strip.astype(np.uint8)


class HUD:
    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    MONO  = cv2.FONT_HERSHEY_DUPLEX
    CYAN  = (0, 230, 255)
    WHITE = (220, 220, 220)
    GREEN = (0, 255, 140)
    DIM   = (80,  80,  80)

    def __init__(self, h, w, dev_str):
        self.h, self.w = h, w
        self.dev_str   = dev_str
        self._fps_buf  = collections.deque(maxlen=30)
        self._last_t   = time.time()

    def tick(self):
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._last_t, 1e-6))
        self._last_t = now

    @property
    def fps(self):
        return np.mean(self._fps_buf) if self._fps_buf else 0.0

    def draw(self, frame, portal, info: dict):
        h, w       = self.h, self.w
        alpha      = portal._alpha
        hand_count = info["hand_count"]
        touching   = info["fingers_touching"]

        frame[0:48, :] = (frame[0:48, :] * 0.35).astype(np.uint8)
        cv2.putText(frame, "Ghost / Invisibility Mode  by Tuba",
                    (16, 30), self.FONT, 0.60, self.CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS {self.fps:05.1f}",
                    (w-160, 30), self.FONT, 0.52, self.GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"[ {self.dev_str} ]",
                    (w-310, 30), self.FONT, 0.42, self.WHITE, 1, cv2.LINE_AA)

        if alpha > 0.05:
            pulse = int(abs(np.sin(time.time() * 5)) * 80 + 175)
            cv2.circle(frame, (w-20, 12), 7, (0, pulse, 255), -1)
            cv2.putText(frame, "GHOST ACTIVE",
                        (w-185, 16), self.FONT, 0.38, (0,230,255), 1, cv2.LINE_AA)
        elif portal.active:
            cv2.circle(frame, (w-20, 12), 7, (0,255,80), -1)
            cv2.putText(frame, "PORTAL READY  — PINCH TO VANISH",
                        (w-340, 16), self.FONT, 0.38, (0,255,80), 1, cv2.LINE_AA)

        frame[h-46:h, :] = (frame[h-46:h, :] * 0.35).astype(np.uint8)
        bx, by, bw2 = 16, h-28, 160
        cv2.rectangle(frame, (bx, by), (bx+bw2, by+10), (40,40,40), -1)
        filled = int(bw2 * alpha)
        if filled > 0:
            cv2.rectangle(frame, (bx, by), (bx+filled, by+10),
                          (int(80*(1-alpha)), int(200*alpha), 255), -1)
        cv2.putText(frame, f"GHOST {int(alpha*100):3d}%",
                    (bx, by-4), self.FONT, 0.40, self.CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"HANDS  {hand_count}",
                    (bx+200, by+8), self.FONT, 0.40, self.WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, time.strftime("%H:%M:%S"),
                    (w-110, by+8), self.MONO, 0.42, self.DIM, 1, cv2.LINE_AA)

        if touching:
            tw = cv2.getTextSize("PINCH DETECTED!", self.FONT, 0.55, 2)[0][0]
            cv2.putText(frame, "PINCH DETECTED!", ((w-tw)//2, h-52),
                        self.FONT, 0.55, (0,255,120), 2, cv2.LINE_AA)
        elif alpha < 0.3 and not portal.active:
            if hand_count == 0:
                hint = "SHOW BOTH HANDS  ->  SPREAD APART  ->  PINCH TO VANISH"
            else:
                hint = "PINCH THUMB + INDEX TOGETHER ON EITHER HAND"
            tw = cv2.getTextSize(hint, self.FONT, 0.40, 1)[0][0]
            cv2.putText(frame, hint, ((w-tw)//2, h-52),
                        self.FONT, 0.40, (120,120,120), 1, cv2.LINE_AA)

        return frame