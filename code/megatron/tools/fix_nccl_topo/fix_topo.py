import argparse
import xml.etree.ElementTree as ET
import copy
import sys
import os
import subprocess
import tempfile

def indent(elem, level=0):
    """
    In-place prettyprint formatter for ElementTree.
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def generate_topology(gen_cmd, temp_xml_path):
    """
    Runs the generation command with NCCL_TOPO_DUMP_FILE set.
    """
    print(f"No input file provided. Generating topology...")
    print(f"Command: {gen_cmd}")
    print(f"Dump Target: {temp_xml_path}")

    # Copy current environment and set the dump file variable
    env = os.environ.copy()
    env["NCCL_TOPO_DUMP_FILE"] = temp_xml_path
    env["ACCL_DISABLE"] = "1"

    try:
        # Run the command. Shell=True allows passing complex strings like "mpirun -np 8 ..."
        subprocess.run(gen_cmd, shell=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\nError: The generation command failed with return code {e.returncode}.")
        print("Please ensure 'nccl-tests' are installed or your custom command is correct.")
        sys.exit(1)
    
    if not os.path.exists(temp_xml_path):
        print(f"Error: The command ran, but '{temp_xml_path}' was not created.")
        print("Ensure the command actually initializes NCCL.")
        sys.exit(1)
        
    print("Topology successfully generated.\n")

def fix_topology(input_file, output_file):
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # 1. Find the source <nvlink> element to use as a template
    source_nvlink = None
    for gpu in root.iter('gpu'):
        nvlink = gpu.find('nvlink')
        if nvlink is not None:
            source_nvlink = nvlink
            break
    
    if source_nvlink is None:
        print("Error: No existing <nvlink> tag found in any GPU to copy from.")
        sys.exit(1)

    print(f"Found template nvlink tag inside GPU rank {source_nvlink.get('target', 'unknown')}")

    # 2. Iterate through all GPUs and inject the nvlink tag if missing
    count = 0
    for gpu in root.iter('gpu'):
        if gpu.find('nvlink') is None:
            new_nvlink = copy.deepcopy(source_nvlink)
            gpu.append(new_nvlink)
            count += 1
    
    print(f"Fixed {count} GPUs by inserting nvlink tags.")

    # 3. Apply indentation
    indent(root)

    # 4. Write to output file
    try:
        tree.write(output_file, encoding='UTF-8', xml_declaration=False)
        print(f"Successfully wrote fixed topology to '{output_file}'")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix NCCL topology XML. Generates input automatically if not provided."
    )
    # Output is now the first positional argument
    parser.add_argument("output_file", help="Path to save the fixed output XML file")
    
    # Input is optional
    parser.add_argument("--input", "-i", help="Path to existing input XML. If omitted, triggers generation.", default=None)
    
    # Command to generate topo if input is missing
    # Default is a minimal all_reduce_perf run on 8 GPUs
    num_gpus = os.environ.get("NPROC_PER_NODE", 8)
    print(f"num_gpus: {num_gpus}")

    default_cmd = f"all_reduce_perf -b 8 -e 8 -g {num_gpus}" 
    parser.add_argument("--gen-cmd", help=f"Command to trigger NCCL init. Default: '{default_cmd}'", default=default_cmd)

    args = parser.parse_args()

    input_path = args.input
    is_temp = False

    # Logic: If no input file, generate one
    if input_path is None:
        # Create a temp file path
        fd, temp_path = tempfile.mkstemp(suffix=".xml", prefix="nccl_topo_dump_")
        os.close(fd) # Close file descriptor, we just need the path
        
        generate_topology(args.gen_cmd, temp_path)
        input_path = temp_path
        is_temp = True

    # Run the fix
    fix_topology(input_path, args.output_file)

    # Cleanup temp file
    if is_temp:
        try:
            os.remove(input_path)
            print(f"Cleaned up temporary file: {input_path}")
        except OSError:
            pass