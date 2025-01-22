import random
import csv

# Define tasks and personnel requirements
tasks = [
    {"task_ID": "T-001", "task_Name": "Aircraft Marshalling", "personnel_per_gate": 1},
    {"task_ID": "T-002", "task_Name": "Passenger Assistance", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-003", "task_Name": "Loading and Unloading Baggage", "personnel_per_gate": random.randint(2, 3)},
    {"task_ID": "T-004", "task_Name": "Baggage Sorting", "personnel_per_gate": random.randint(2, 4)},
    {"task_ID": "T-005", "task_Name": "Cargo Management", "personnel_per_gate": 2},
    {"task_ID": "T-006", "task_Name": "Refueling", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-007", "task_Name": "Cleaning", "personnel_per_gate": random.randint(2, 3)},
    {"task_ID": "T-008", "task_Name": "De-icing", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-009", "task_Name": "Water and Waste Services", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-010", "task_Name": "Aircraft Inspections", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-011", "task_Name": "Ground Power Unit Operations", "personnel_per_gate": 1},
    {"task_ID": "T-012", "task_Name": "Safety Zone Maintenance", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-013", "task_Name": "Pushback and Towing", "personnel_per_gate": random.randint(2, 3)},
    {"task_ID": "T-014", "task_Name": "Conveyor Belts and Baggage Carts", "personnel_per_gate": random.randint(2, 4)},
    {"task_ID": "T-015", "task_Name": "Ramp Equipment Management", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-016", "task_Name": "Liaison with ATC", "personnel_per_gate": 1},
    {"task_ID": "T-017", "task_Name": "Coordination with Airlines", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-018", "task_Name": "Emergency Response", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-019", "task_Name": "Weather Adaptations", "personnel_per_gate": random.randint(1, 2)}
]

# Define gates and floors
upper_floor_gates = list(range(1, 3)) + list(range(12, 19)) + list(range(28, 44))  # Upper floor gates
lower_floor_gates = list(range(19, 26))  # Lower floor gates

def generate_crew_data():
    data = []
    crew_id = 1

    # Assign crew to tasks on the ground floor
    for shift in range(1, 5):
        for task in tasks:
            num_crew = random.randint(10, 15)  # Random number of ground floor personnel
            for _ in range(num_crew):
                crew_data = {
                    "Crew_ID": f"C-{crew_id:03}",
                    "task_ID": task["task_ID"],
                    "task_Name": task["task_Name"],
                    "Floor_No": 0,  # Ground floor
                    "Gate_number": 0,  # No gates on ground floor
                    "Shift_No": shift
                }
                data.append(crew_data)
                crew_id += 1

    # Assign crew to tasks on the upper and lower floors
    for shift in range(1, 5):
        for gate in upper_floor_gates + lower_floor_gates:
            floor_no = 2 if gate in upper_floor_gates else 1  # Upper floor = 2, Lower floor = 1
            for task in tasks:
                num_crew = task["personnel_per_gate"]  # Personnel required for the task
                for _ in range(num_crew):
                    crew_data = {
                        "Crew_ID": f"C-{crew_id:03}",
                        "task_ID": task["task_ID"],
                        "task_Name": task["task_Name"],
                        "Floor_No": floor_no,
                        "Gate_number": gate,
                        "Shift_No": shift
                    }
                    data.append(crew_data)
                    crew_id += 1

    return data

# Generate the dataset
data = generate_crew_data()

# Save to a CSV file
output_file = "crew_data.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Crew_ID", "task_ID", "task_Name", "Floor_No", "Gate_number", "Shift_No"])
    writer.writeheader()
    writer.writerows(data)

print(f"Synthetic dataset saved to {output_file}")