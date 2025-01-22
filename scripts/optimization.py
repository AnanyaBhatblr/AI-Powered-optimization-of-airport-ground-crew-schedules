# scripts/optimization.py
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

def optimize_schedule(tasks_file, crew_file):
    tasks = pd.read_csv(tasks_file)
    crew = pd.read_csv(crew_file)

    task_ids = tasks['Task ID'].tolist()
    crew_ids = crew['Crew ID'].tolist()
    x = {(t, c): LpVariable(f"x_{t}_{c}", cat="Binary") for t in task_ids for c in crew_ids}

    prob = LpProblem("Task_Allocation", LpMinimize)
    prob += lpSum(x[t, c] for t in task_ids for c in crew_ids)

    for t in task_ids:
        prob += lpSum(x[t, c] for c in crew_ids) == 1

    prob.solve()

    allocations = []
    for t in task_ids:
        for c in crew_ids:
            if x[t, c].varValue:
                allocations.append((t, c))

    allocations_df = pd.DataFrame(allocations, columns=['Task ID', 'Crew ID'])
    allocations_df.to_csv("../data/task_allocations.csv", index=False)
    print("Optimization complete!")

if __name__ == "__main__":
    optimize_schedule("../data/processed_tasks.csv", "../data/processed_crew.csv")
