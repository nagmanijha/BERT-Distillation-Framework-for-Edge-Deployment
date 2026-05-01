def calculate_accuracy_retention(teacher_acc: float, student_acc: float) -> float:
    """
    Calculate the accuracy retention of the student model relative to the teacher.
    """
    if teacher_acc == 0:
        return 0.0
    relative_accuracy = (student_acc / teacher_acc) * 100.0
    print(f"Student retained {relative_accuracy:.2f}% of Teacher's accuracy.")
    return relative_accuracy

def run_accuracy_benchmark():
    teacher_accuracy = 0.925
    student_accuracy = 0.891
    retention = calculate_accuracy_retention(teacher_accuracy, student_accuracy)
    assert retention > 90.0, "Accuracy retention is below 90% threshold!"

if __name__ == "__main__":
    run_accuracy_benchmark()
