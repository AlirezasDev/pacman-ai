# Pacman AI Assignment

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the agents

Run the interactive menu:

```powershell
python student_assignment.py
```

Run the assessment for all three agents:

```powershell
python student_assignment.py --assess
```

The assessment asks for the number of episodes per agent. The default is 5.

Run the human game directly:

```powershell
python game.py
```

The per-move timeout in the assignment wrapper is 5 seconds. The minimax and alpha-beta searches use bounded depth and an internal deadline; MCTS also stops its iteration loop before the deadline.
