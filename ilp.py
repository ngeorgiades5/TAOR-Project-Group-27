import gurobipy as gp
from gurobipy import GRB
import itertools
import pandas as pd

def schedule_double_round_robin(n_teams, n_periods, alpha=1, beta=1):
    teams = list(range(1, n_teams + 1))
    rounds = 2 * (n_teams - 1)
    games = list(itertools.permutations(teams, 2))  # Home/away pairs
    n_games = len(games)
    periods = list(range(n_periods))

    # Precompute play, first, second parameters
    play = {(g, t): 0 for g in range(n_games) for t in teams}
    first = {(g, t): 0 for g in range(n_games) for t in teams}
    second = {(g, t): 0 for g in range(n_games) for t in teams}

    for g_idx, (t1, t2) in enumerate(games):
        play[g_idx, t1] = 1
        play[g_idx, t2] = 1
        first[g_idx, t1] = 1  # Home team
        second[g_idx, t2] = 1  # Away team

    games_per_period = n_games // (rounds * n_periods)

    model = gp.Model("DoubleRoundRobin_ILP1")
    model.Params.TimeLimit = 3600

    # Decision Variables
    x = model.addVars(n_games, rounds, n_periods, vtype=GRB.BINARY, name="x")
    q = model.addVars(n_games, rounds, vtype=GRB.BINARY, name="q")
    d = model.addVars(n_games, vtype=GRB.CONTINUOUS, name="d")
    e = model.addVars(n_games, vtype=GRB.CONTINUOUS, name="e")
    y = model.addVars(n_games, vtype=GRB.BINARY, name="y")

    # New variables for breaks and break mismatches
    b = model.addVars(teams, rounds, vtype=GRB.BINARY, name="b")  # Break indicator
    z = model.addVars(n_games, vtype=GRB.BINARY, name="z")        # Break mismatch

    # Track home/away assignments
    h = model.addVars(teams, rounds, vtype=GRB.BINARY, name="h")  # 1 if home
    a = model.addVars(teams, rounds, vtype=GRB.BINARY, name="a")  # 1 if away

    # Objective: Minimize combined mismatches
    model.setObjective(
        alpha * y.sum() + beta * z.sum(), 
        GRB.MINIMIZE
    )

    # Constraints
    # C1: Each game is scheduled exactly once
    for g in range(n_games):
        model.addConstr(x.sum(g, '*', '*') == 1)

    # C2: Each team plays once per round (home or away)
    for r in range(rounds):
        for t in teams:
            model.addConstr(
                gp.quicksum(
                    play[g, t] * x[g, r, p] 
                    for g in range(n_games) 
                    for p in periods
                ) == 1
            )

    # C3: Fixed number of games per period
    for r in range(rounds):
        for p in periods:
            model.addConstr(x.sum('*', r, p) == games_per_period)

    # C4: Link x and q (q[g,r]=1 if game g is in round r)
    for g in range(n_games):
        for r in range(rounds):
            model.addConstr(x.sum(g, r, '*') <= q[g, r])

    # C5-C6: Rest difference calculation
    M = n_periods
    for g in range(n_games):
        t_home, t_away = games[g]
        for r in range(1, rounds):
            prev_period_home = gp.quicksum(
                (p + 1) * play[g_prev, t_home] * x[g_prev, r-1, p]
                for g_prev in range(n_games)
                for p in periods
            )
            prev_period_away = gp.quicksum(
                (p + 1) * play[g_prev, t_away] * x[g_prev, r-1, p]
                for g_prev in range(n_games)
                for p in periods
            )
            model.addConstr(
                (prev_period_away - prev_period_home) <= 
                d[g] - e[g] + M * (1 - q[g, r])
            )
            model.addConstr(
                (prev_period_home - prev_period_away) <= 
                e[g] - d[g] + M * (1 - q[g, r])
            )

    # C7: Mismatch indicator
    for g in range(n_games):
        model.addConstr(d[g] + e[g] <= M * y[g])

    # --- New constraints for breaks ---
    # C8: Home/away assignment
    for t in teams:
        for r in range(rounds):
            model.addConstr(
                h[t, r] == gp.quicksum(
                    first[g, t] * x[g, r, p] 
                    for g in range(n_games) 
                    for p in periods
                )
            )
            model.addConstr(
                a[t, r] == gp.quicksum(
                    second[g, t] * x[g, r, p] 
                    for g in range(n_games) 
                    for p in periods
                )
            )

    # C9: Break detection (consecutive home/away)
    for t in teams:
        for r in range(1, rounds):
            model.addConstr(
                b[t, r] >= h[t, r] + h[t, r-1] - 1
            )
            model.addConstr(
                b[t, r] >= a[t, r] + a[t, r-1] - 1
            )

    # C10: Break mismatch calculation
    M_break = rounds
    for g in range(n_games):
        t_home, t_away = games[g]
        for r in range(1, rounds):
            total_breaks_home = gp.quicksum(b[t_home, k] for k in range(1, r))
            total_breaks_away = gp.quicksum(b[t_away, k] for k in range(1, r))
            model.addConstr(
                total_breaks_home - total_breaks_away <= M_break * z[g]
            )
            model.addConstr(
                total_breaks_away - total_breaks_home <= M_break * z[g]
            )

    model.optimize()

    # Extract schedule and rest days
    schedule = {}
    rest_days_dict = {}
    breaks_dict = {}

    if model.status == GRB.OPTIMAL:
        for r in range(rounds):
            for p in periods:
                games_in_period = [
                    games[g] for g in range(n_games) 
                    if x[g, r, p].X > 0.5
                ]
                schedule[(r, p)] = games_in_period

        # Extract rest days and breaks
        for t in teams:
            rest_days_dict[t] = {}
            breaks_dict[t] = {}
            for r in range(rounds):
                if r == 0:
                    rest_days_dict[t][r] = "First round"
                else:
                    rest_days_dict[t][r] = sum(
                        1 for k in range(1, r) if b[t, k].X > 0.5
                    )
                breaks_dict[t][r] = b[t, r].X

        return schedule, rest_days_dict, breaks_dict, model.objVal
    else:
        return None, None, None, None

if __name__ == '__main__':
  
  n_teams = 8
  n_periods = 2
  schedule, rest_days, breaks, mismatches = schedule_double_round_robin(n_teams, n_periods)
  
  # Convert the extracted schedule into a DataFrame with Rounds as the first column
  if schedule:
      # Initialize DataFrame structure
      df_dict = {"Round": []}
      for p in range(n_periods):
          df_dict[f"Period {p+1}"] = []
  
      # Populate the dictionary with match information
      for r in range(n_periods * (n_teams - 1)):  # Total rounds
          df_dict["Round"].append(f"Round {r+1}")
          for p in range(n_periods):
              match_info = ", ".join([f"{t1}-{t2}" for t1, t2 in schedule.get((r, p), [])])
              df_dict[f"Period {p+1}"].append(match_info)
  
      # Convert to DataFrame
      df_schedule = pd.DataFrame(df_dict)
      df_schedule.to_excel("Optimized_Schedule_ILP_8_teams.xlsx", index=False)
      print(df_schedule)
      # Initialize a dictionary to track match occurrences between teams
      team_vs_team_count = {t1: {t2: 0 for t2 in range(1, 9) if t2 != t1} for t1 in range(1, 9)}
      
      # Populate the counts from df_schedule
      for _, row in df_schedule.iterrows():
          for p in range(1, 3):  # Two periods
              match_info = row[f"Period {p}"]
              if match_info:
                  matches = match_info.split(", ")
                  for match in matches:
                      if "-" in match:
                          t1, t2 = map(int, match.split("-"))
                          team_vs_team_count[t1][t2] += 1
                          team_vs_team_count[t2][t1] += 1
  
      # Check for violations: each team should have played against every other team exactly twice
      violations = []
      for t1 in range(1, 9):
          for t2 in range(1, 9):
              if t1 != t2 and team_vs_team_count[t1][t2] != 2:
                  violations.append((t1, t2, team_vs_team_count[t1][t2]))
      
      # Convert violations to a DataFrame for display
      if violations:
          df_violations = pd.DataFrame(violations, columns=["Team 1", "Team 2", "Matches Played"])
          import ace_tools as tools
          tools.display_dataframe_to_user(name="Match Count Violations", dataframe=df_violations)
      else:
          print("The schedule is valid: each team has played against every other team exactly twice.")
  
  else:
      print("No schedule available.")
