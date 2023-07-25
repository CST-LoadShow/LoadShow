def function(file_path: str):
    # cpu_file: f"{file_path}/cpu.csv"
    # gpu_file: f"{file_path}/gpu.csv"
    
    program_name = "program_name" # result of random forest classifier
    with open(f"{file_path}/log", "w") as f:
        f.write("confusion matrix")
    return program_name
