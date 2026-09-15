"""Dice tool the DM LLM can call for chance-based outcomes."""
import random
import re

NOTATION_RE = re.compile(r"^\s*(\d{1,3})d(\d{1,4})([+-]\d+)?\s*$", re.IGNORECASE)


def parse_notation(notation: str) -> tuple[int, int, int]:
    m = NOTATION_RE.match(notation or "")
    if not m:
        raise ValueError(f"Invalid dice notation '{notation}'. Use NdM+K, e.g. 1d20, 2d6+3.")
    n, sides, mod = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if not (1 <= n <= 100):
        raise ValueError("Roll 1-100 dice.")
    if sides not in (2, 3, 4, 6, 8, 10, 12, 20, 100):
        raise ValueError("Sides must be one of d2,d3,d4,d6,d8,d10,d12,d20,d100.")
    if not (-1000 <= mod <= 1000):
        raise ValueError("Modifier out of range.")
    return n, sides, mod


def roll_dice_notation(notation: str) -> dict:
    """Roll dice. Returns individual rolls, modifier, and total."""
    n, sides, mod = parse_notation(notation)
    rolls = [random.randint(1, sides) for _ in range(n)]
    return {
        "notation": f"{n}d{sides}{f'{mod:+d}' if mod else ''}",
        "rolls": rolls,
        "modifier": mod,
        "total": sum(rolls) + mod,
    }


try:
    from langchain_core.tools import tool

    @tool
    def roll_dice(notation: str) -> str:
        """Roll tabletop dice. Input like '1d20', '2d6+3', '4d6'. Use whenever an action has a chance-based outcome (attacks, checks, saves)."""
        try:
            r = roll_dice_notation(notation)
            return f"Rolled {r['notation']}: [{', '.join(map(str, r['rolls']))}] + ({r['modifier']}) = {r['total']}"
        except ValueError as e:
            return f"Dice error: {e}"
except ImportError:  # langchain_core missing (shouldn't happen) — plain fallback
    def roll_dice(notation: str) -> str:  # type: ignore
        try:
            r = roll_dice_notation(notation)
            return f"Rolled {r['notation']}: [{', '.join(map(str, r['rolls']))}] + ({r['modifier']}) = {r['total']}"
        except ValueError as e:
            return f"Dice error: {e}"
