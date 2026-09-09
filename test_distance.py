from src.safety.distance_calculator import (
    calculate_center_distance,
    calculate_ground_distance,
    find_closest_machine,
    calculate_person_machine_distances
)


# ---------------------------------------------------------
# TEST 1: Distance between two bounding boxes
# ---------------------------------------------------------

person_bbox = [100, 100, 200, 400]

machine_bbox = [300, 200, 600, 500]

center_distance = calculate_center_distance(
    person_bbox,
    machine_bbox
)

ground_distance = calculate_ground_distance(
    person_bbox,
    machine_bbox
)

print("=" * 60)
print("DISTANCE CALCULATION TEST")
print("=" * 60)

print(f"\nPerson bbox  : {person_bbox}")
print(f"Machine bbox : {machine_bbox}")

print(f"\nCenter distance : {center_distance:.2f} pixels")
print(f"Ground distance: {ground_distance:.2f} pixels")


# ---------------------------------------------------------
# TEST 2: Find closest machine
# ---------------------------------------------------------

machines = [
    {
        "class_name": "Excavator",
        "confidence": 0.92,
        "bbox": [300, 200, 600, 500]
    },
    {
        "class_name": "Dump Truck",
        "confidence": 0.87,
        "bbox": [800, 250, 1200, 600]
    }
]

closest_machine = find_closest_machine(
    person_bbox,
    machines
)

print("\n" + "=" * 60)
print("CLOSEST MACHINE")
print("=" * 60)

if closest_machine:
    print(f"\nMachine   : {closest_machine['class_name']}")
    print(f"Confidence: {closest_machine['confidence']:.2f}")
    print(f"Distance  : {closest_machine['distance']:.2f} pixels")
else:
    print("\nNo machine found.")


# ---------------------------------------------------------
# TEST 3: Multiple tracked persons
# ---------------------------------------------------------

persons = [
    {
        "track_id": 1,
        "bbox": [100, 100, 200, 400],
        "confidence": 0.91
    },
    {
        "track_id": 2,
        "bbox": [650, 150, 750, 450],
        "confidence": 0.88
    }
]

results = calculate_person_machine_distances(
    persons,
    machines
)

print("\n" + "=" * 60)
print("PERSON → CLOSEST MACHINE")
print("=" * 60)

for result in results:

    print(f"\nPerson ID: {result['track_id']}")

    if result["machine_class"]:

        print(f"Closest machine: {result['machine_class']}")
        print(
            f"Distance: "
            f"{result['distance']:.2f} pixels"
        )

    else:

        print("No machine detected nearby.")


print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)