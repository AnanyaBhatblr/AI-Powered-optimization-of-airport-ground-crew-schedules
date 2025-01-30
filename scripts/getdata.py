import random
import csv

# Define tasks and personnel requirements
tasks = [
    {"task_ID": "T-001", "task_Name": "Aircraft Marshalling", "personnel_per_gate": 1},
    {"task_ID": "T-002", "task_Name": "Passenger Assistance", "personnel_per_gate": random.randint(1, 3)},
    {"task_ID": "T-003", "task_Name": "Loading and Unloading Baggage", "personnel_per_gate": random.randint(2, 4)},
    {"task_ID": "T-004", "task_Name": "Baggage Sorting", "personnel_per_gate": random.randint(2, 5)},
    {"task_ID": "T-005", "task_Name": "Cargo Management", "personnel_per_gate": 3},
    {"task_ID": "T-006", "task_Name": "Refueling", "personnel_per_gate": random.randint(1, 3)},
    {"task_ID": "T-007", "task_Name": "Cleaning", "personnel_per_gate": random.randint(2, 4)},
    {"task_ID": "T-008", "task_Name": "De-icing", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-009", "task_Name": "Water and Waste Services", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-010", "task_Name": "Aircraft Inspections", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-011", "task_Name": "Ground Power Unit Operations", "personnel_per_gate": 1},
    {"task_ID": "T-012", "task_Name": "Safety Zone Maintenance", "personnel_per_gate": random.randint(1, 3)},
    {"task_ID": "T-013", "task_Name": "Pushback and Towing", "personnel_per_gate": random.randint(2, 4)},
    {"task_ID": "T-014", "task_Name": "Ramp Equipment Management", "personnel_per_gate": random.randint(1, 3)},
    {"task_ID": "T-015", "task_Name": "Liaison with ATC", "personnel_per_gate": 1},
    {"task_ID": "T-016", "task_Name": "Coordination with Airlines", "personnel_per_gate": random.randint(1, 2)},
    {"task_ID": "T-017", "task_Name": "Emergency Response", "personnel_per_gate": random.randint(1, 2)}
]

# Define gates and floor levels for Hyderabad Airport (RGIA)
gates_arrival = list(range(1, 13))  # Arrivals area (Level 0)
gates_international = list(range(13, 25))  # International Departures (Level 1)
gates_domestic = list(range(25, 43))  # Domestic Departures (Level 1)
gates_admin = list(range(43, 56))  # Administrative & Operations (Levels 2-5)

def generate_crew_data():
    data = []
    crew_id = 1

    # Assign crew to tasks on Level 0 (Arrivals area)
    for shift in range(1, 5):
        for task in tasks:
            num_crew = random.randint(8, 12)
            for _ in range(num_crew):
                data.append({
                    "Crew_ID": f"C-{crew_id:03}",
                    "task_ID": task["task_ID"],
                    "task_Name": task["task_Name"],
                    "Floor_No": 0,
                    "Gate_number": 0,
                    "Shift_No": shift
                })
                crew_id += 1

    # Assign crew to tasks on other levels
    for shift in range(1, 5):
        for gate in gates_arrival + gates_international + gates_domestic + gates_admin:
            if gate in gates_arrival:
                floor_no = 0
            elif gate in gates_international or gate in gates_domestic:
                floor_no = 1
            else:
                floor_no = random.randint(2, 5)
            
            for task in tasks:
                num_crew = task["personnel_per_gate"]
                for _ in range(num_crew):
                    data.append({
                        "Crew_ID": f"C-{crew_id:03}",
                        "task_ID": task["task_ID"],
                        "task_Name": task["task_Name"],
                        "Floor_No": floor_no,
                        "Gate_number": gate,
                        "Shift_No": shift
                    })
                    crew_id += 1

    return data

# Generate the dataset
data = generate_crew_data()


# Save to a CSV file
output_file = "../dataset/HT1/crew_dataHT1.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Crew_ID", "task_ID", "task_Name", "Floor_No", "Gate_number", "Shift_No"])
    writer.writeheader()
    writer.writerows(data)

print(f"Synthetic dataset saved to {output_file}")