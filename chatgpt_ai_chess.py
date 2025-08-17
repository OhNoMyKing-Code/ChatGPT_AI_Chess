#!/usr/bin/env python3
"""
ChatGPT-AI-Chess — a compact UCI chess engine in Python.
- Protocol: UCI
- Search: Iterative deepening + Negamax alpha-beta + Quiescence
- Heuristics: Transposition table (Zobrist via python-chess), move ordering
  (PV move, captures MVV-LVA, killer/history), simple time management
- Eval: material + piece-square tables + mobility + basic king safety

Dependencies: python-chess
    pip install python-chess

Run locally:
    python chatgpt_ai_chess.py

Use in a GUI (e.g., CuteChess/Arena) by adding this script as a UCI engine.
"""
import sys
import time
import math
import random
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import chess
import chess.polyglot

# -------------------------- Utility / Types --------------------------
INF = 10_000_000
MATE_SCORE = 9_000_000
MATE_IN_MAX = 10_000

@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: int   # 0=EXACT, 1=LOWERBOUND, 2=UPPERBOUND
    move: Optional[chess.Move]

# -------------------------- Engine Options --------------------------
class Options:
    def __init__(self):
        self.Hash = 64  # MB
        self.Threads = 1
        self.Skill = 15  # 0..20 reduces depth via time budget
        self.OwnBook = False

OPTIONS = Options()

# -------------------------- Piece-Square Tables --------------------------
# Simplified PSTs (middlegame) from common sources; tuned lightly.
# Indexed from white POV; we'll mirror for black via 63 - sq.
P = [
    0,  5,  5, -5, -5,  5, 10,  0,
    0,  5, -5,  0,  0, -5,  5,  0,
    0,  5, 10, 15, 15, 10,  5,  0,
    5, 10, 15, 20, 20, 15, 10,  5,
    5, 10, 15, 20, 20, 15, 10,  5,
    0,  5, 10, 15, 15, 10,  5,  0,
    0,  5, -5,  0,  0, -5,  5,  0,
    0,  5,  5, -5, -5,  5, 10,  0,
]
N = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
B = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -10,  5, 10, 10, 10, 10,  5,-10,
    -10,  5, 10, 15, 15, 10,  5,-10,
    -10,  5, 10, 15, 15, 10,  5,-10,
    -10,  5, 10, 10, 10, 10,  5,-10,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
R = [
     0,  0,  5, 10, 10,  5,  0,  0,
     0,  5, 10, 15, 15, 10,  5,  0,
     0,  0,  5, 10, 10,  5,  0,  0,
     0,  0,  5, 10, 10,  5,  0,  0,
     0,  0,  5, 10, 10,  5,  0,  0,
     0,  0,  5, 10, 10,  5,  0,  0,
     5,  5, 10, 15, 15, 10,  5,  5,
     0,  0,  5, 10, 10,  5,  0,  0,
]
Q = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5, 10, 10,  5,  0, -5,
     -5,  0,  5, 10, 10,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
K_MID = [
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PST = {
    chess.PAWN: P,
    chess.KNIGHT: N,
    chess.BISHOP: B,
    chess.ROOK: R,
    chess.QUEEN: Q,
    chess.KING: K_MID,
}

# -------------------------- Evaluation --------------------------

def pst_score(piece_type: chess.PieceType, sq: chess.Square, color: chess.Color) -> int:
    arr = PST[piece_type]
    idx = sq if color == chess.WHITE else chess.square_mirror(sq)
    return arr[idx]


def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -MATE_SCORE + (0 if board.turn else 1)  # mate for side to move
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0

    score = 0

    # Material + PST
    for piece_type in PIECE_VALUES:
        wp = board.pieces(piece_type, chess.WHITE)
        bp = board.pieces(piece_type, chess.BLACK)
        score += PIECE_VALUES[piece_type] * (len(wp) - len(bp))
        for sq in wp:
            score += pst_score(piece_type, sq, chess.WHITE)
        for sq in bp:
            score -= pst_score(piece_type, sq, chess.BLACK)

    # Mobility (small)
    score += 2 * (len(list(board.legal_moves)) if board.turn == chess.WHITE else 0)
    board.push(chess.Move.null())
    score -= 2 * (len(list(board.legal_moves)) if board.turn == chess.BLACK else 0)
    board.pop()

    # King safety (very rough): penalty for open files next to king
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            file_idx = chess.square_file(king_sq)
            penalty = 0
            for f in [max(0, file_idx - 1), file_idx, min(7, file_idx + 1)]:
                file_mask = [chess.square(f, r) for r in range(8)]
                pawns = board.pieces(chess.PAWN, color)
                cover = any(sq in pawns for sq in file_mask)
                if not cover:
                    penalty += 8
            if color == chess.WHITE:
                score -= penalty
            else:
                score += penalty

    return score if board.turn == chess.WHITE else -score

# -------------------------- Search --------------------------
class Searcher:
    def __init__(self):
        self.tt: Dict[int, TTEntry] = {}
        self.killer: Dict[int, List[chess.Move]] = {}
        self.history: Dict[Tuple[int, int], int] = {}
        self.stop = False
        self.best_move: Optional[chess.Move] = None
        self.uci_info_lock = threading.Lock()
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit_ms = 3000

    # Move ordering helpers
    def mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square) or 0
            attacker = board.piece_type_at(move.from_square) or 0
            return 10_000 + 10 * PIECE_VALUES.get(victim, 0) - PIECE_VALUES.get(attacker, 0)
        return 0

    def killer_score(self, ply: int, move: chess.Move) -> int:
        if ply in self.killer and move in self.killer[ply]:
            return 5_000
        return 0

    def history_score(self, move: chess.Move, side: int) -> int:
        return self.history.get((move.from_square, move.to_square), 0)

    def ordered_moves(self, board: chess.Board, ply: int, tt_move: Optional[chess.Move]) -> List[chess.Move]:
        moves = list(board.legal_moves)
        def key(m: chess.Move):
            sc = 0
            if tt_move and m == tt_move: sc += 50_000
            sc += self.mvv_lva_score(board, m)
            sc += self.killer_score(ply, m)
            sc += self.history_score(m, board.turn)
            return sc
        moves.sort(key=key, reverse=True)
        return moves

    # Quiescence search to resolve horizon effect on captures
    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        # Only consider captures (and checking moves if desired)
        for move in list(board.legal_moves):
            if not board.is_capture(move):
                continue
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def probe_tt(self, key: int, depth: int, alpha: int, beta: int) -> Tuple[Optional[int], Optional[chess.Move]]:
        entry = self.tt.get(key)
        if entry and entry.depth >= depth:
            if entry.flag == 0:
                return entry.score, entry.move
            if entry.flag == 1 and entry.score > alpha:
                return entry.score, entry.move
            if entry.flag == 2 and entry.score < beta:
                return entry.score, entry.move
        return None, entry.move if entry else None

    def store_tt(self, key: int, depth: int, score: int, flag: int, move: Optional[chess.Move]):
        self.tt[key] = TTEntry(key, depth, score, flag, move)

    def should_stop(self) -> bool:
        if self.time_limit_ms <= 0:
            return False
        return (time.time() - self.start_time) * 1000.0 >= self.time_limit_ms

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        if self.should_stop():
            return evaluate(board)
        self.nodes += 1

        key = chess.polyglot.zobrist_hash(board)
        tt_score, tt_move = self.probe_tt(key, depth, alpha, beta)
        if tt_score is not None:
            return tt_score

        if depth == 0:
            return self.quiescence(board, alpha, beta, ply)

        legal = list(board.legal_moves)
        if not legal:
            # stalemate or checkmate handled in eval path
            return evaluate(board)

        best_move = None
        value = -INF
        for move in self.ordered_moves(board, ply, tt_move):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > value:
                value = score
                best_move = move
            if value > alpha:
                alpha = value
                # history update on PV move
                self.history[(move.from_square, move.to_square)] = self.history.get((move.from_square, move.to_square), 0) + depth * depth
            if alpha >= beta:
                # beta-cutoff -> killer move
                if ply not in self.killer:
                    self.killer[ply] = []
                if move not in self.killer[ply]:
                    self.killer[ply].append(move)
                    if len(self.killer[ply]) > 2:
                        self.killer[ply] = self.killer[ply][-2:]
                break

        # store in TT
        flag = 0 if value > alpha and value < beta else (1 if value >= beta else 2)
        self.store_tt(key, depth, value, flag, best_move)
        if ply == 0 and best_move is not None:
            self.best_move = best_move
        return value

    def search(self, board: chess.Board, max_time_ms: int, max_depth: int = 64) -> chess.Move:
        self.start_time = time.time()
        self.time_limit_ms = max_time_ms
        self.stop = False
        self.best_move = None
        self.nodes = 0

        # Iterative deepening
        alpha, beta = -INF, INF
        for depth in range(1, max_depth + 1):
            score = self.negamax(board, depth, alpha, beta, 0)
            if self.stop or self.should_stop():
                break
            bm = self.best_move
            nps = int(self.nodes / max(1e-3, (time.time() - self.start_time)))
            if bm:
                print(f"info depth {depth} score cp {score} time {int((time.time()-self.start_time)*1000)} nodes {self.nodes} nps {nps} pv {bm.uci()}")
            # Aspiration windows (optional): shrink around last score
            alpha, beta = score - 50, score + 50
        return self.best_move or random.choice(list(board.legal_moves))

# -------------------------- Time Management --------------------------

def compute_time_allocation(board: chess.Board, wtime: int, btime: int, winc: int, binc: int) -> int:
    # Allocate a small slice of remaining time + some inc; scale with Skill
    remain = wtime if board.turn == chess.WHITE else btime
    inc = winc if board.turn == chess.WHITE else binc
    base = max(50, remain // 30)  # ~3% of remaining time
    budget = base + inc // 2
    # Skill lowers budget -> weaker/faster
    reduction = (20 - OPTIONS.Skill) / 20.0
    budget = int(budget * (0.6 + 0.4 * reduction))
    return max(50, min(budget, remain - 50))

# -------------------------- Opening Book (optional) -------------------
class LightBook:
    def __init__(self):
        # Minimal hardcoded lines to avoid dumb early mistakes
        self.lines = {
            # e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["Ng1f3"],
        }
    def move(self, board: chess.Board) -> Optional[chess.Move]:
        fen = board.board_fen() + (" w" if board.turn==chess.WHITE else " b")
        for key, sans in self.lines.items():
            if key.startswith(board.board_fen()):
                try:
                    return chess.Move.from_uci(chess.Move.from_san(board, sans[0]).uci())
                except Exception:
                    return None
        return None

BOOK = LightBook()

# -------------------------- UCI Loop --------------------------
class UCI:
    def __init__(self):
        self.board = chess.Board()
        self.searcher = Searcher()
        self.ponder = False
        self.name = "ChatGPT-AI-Chess"
        self.author = "Khanh x ChatGPT"

    def send(self, s: str):
        print(s)
        sys.stdout.flush()

    def uci(self):
        self.send(f"id name {self.name}")
        self.send(f"id author {self.author}")
        self.send("uciok")

    def setoption(self, name: str, value: str):
        if name == "Hash":
            try:
                OPTIONS.Hash = int(value)
            except ValueError:
                pass
        elif name == "Skill":
            try:
                OPTIONS.Skill = max(0, min(20, int(value)))
            except ValueError:
                pass
        elif name == "OwnBook":
            OPTIONS.OwnBook = value.lower() in ["true", "1", "on", "yes"]

    def isready(self):
        self.send("readyok")

    def ucinewgame(self):
        self.board = chess.Board()
        self.searcher = Searcher()

    def position(self, args: List[str]):
        i = 0
        if args[0] == "startpos":
            self.board = chess.Board()
            i = 1
        elif args[0] == "fen":
            fen = " ".join(args[1:7])
            self.board = chess.Board(fen)
            i = 7
        if i < len(args) and args[i] == "moves":
            for mv in args[i+1:]:
                self.board.push_uci(mv)

    def go(self, args: List[str]):
        wtime = btime = winc = binc = movestogo = 0
        depth = 0
        movetime = 0
        i = 0
        while i < len(args):
            if args[i] == "wtime": wtime = int(args[i+1]); i += 2
            elif args[i] == "btime": btime = int(args[i+1]); i += 2
            elif args[i] == "winc": winc = int(args[i+1]); i += 2
            elif args[i] == "binc": binc = int(args[i+1]); i += 2
            elif args[i] == "movestogo": movestogo = int(args[i+1]); i += 2
            elif args[i] == "depth": depth = int(args[i+1]); i += 2
            elif args[i] == "movetime": movetime = int(args[i+1]); i += 2
            else:
                i += 1

        # Opening book move if enabled
        if OPTIONS.OwnBook:
            bm = BOOK.move(self.board)
            if bm:
                self.send(f"bestmove {bm.uci()}")
                return

        alloc = movetime if movetime > 0 else compute_time_allocation(self.board, wtime, btime, winc, binc)
        move = self.searcher.search(self.board, alloc, max_depth=64 if depth==0 else depth)
        self.send(f"bestmove {move.uci()}")

    def loop(self):
        while True:
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                break
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cmd, args = parts[0], parts[1:]

            if cmd == "uci": self.uci()
            elif cmd == "isready": self.isready()
            elif cmd == "setoption":
                # setoption name <Name> value <Value>
                try:
                    name_idx = args.index("name") + 1 if "name" in args else 0
                    value_idx = args.index("value") + 1 if "value" in args else 1
                    name = " ".join(args[name_idx:value_idx-1]) if value_idx>1 else args[0]
                    value = " ".join(args[value_idx:])
                    self.setoption(name, value)
                except Exception:
                    pass
            elif cmd == "ucinewgame": self.ucinewgame()
            elif cmd == "position": self.position(args)
            elif cmd == "go": self.go(args)
            elif cmd == "stop": pass
            elif cmd == "quit": break
            else:
                # ignore unknown
                pass

if __name__ == "__main__":
    UCI().loop()
