import argparse
import os
import shutil

def process_train_file(input_file_path, output_file_path, class_index_map=None, current_class_name=None):
    """
    Processes a train.txt file, adding a third column based on specified logic.
    The output file will have space-separated values.

    Args:
        input_file_path (str): Path to the source train.txt file.
        output_file_path (str): Path to the destination .txt file.
        class_index_map (dict, optional): A dictionary mapping class names to 1-based indices.
                                          Used for train.ALL.txt. Defaults to None.
        current_class_name (str, optional): The name of the specific class being processed (e.g., "PERS", "LOC").
                                            Used for train.CLASS.txt files. Defaults to None.
    """
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t') # Input files are TAB separated
            
            first_two_cols = parts[:2] # Take up to the first two columns
            if len(first_two_cols) < 2:
                # Pad with empty strings if there are fewer than two columns in the input
                outfile.write(' '.join(first_two_cols) + '\n')
                continue

            third_column_value = '0' # Default value for the third column

            # Determine the third column value based on the file type and content
            if class_index_map: # This logic applies to train.ALL.txt
                if len(parts) >= 2 and '-' in parts[1]:
                    extracted_class = parts[1].split('-')[-1] # Remove the leading '-'
                    if extracted_class in class_index_map:
                        third_column_value = str(class_index_map[extracted_class])
            elif current_class_name: # This logic applies to train.CLASS.txt files
                if len(parts) >= 2 and parts[1].endswith(f"-{current_class_name}"):
                    third_column_value = '1'
            # If neither specific case applies, third_column_value remains '0' (for test.txt, train.txt, valid.txt)
            
            # Combine the first two columns (from input) with the determined third column value
            outfile.write(' '.join(first_two_cols + [third_column_value]) + '\n')


def copy_and_transform_data(input_dir, output_dir, output_prefix, output_suffix):
    """
    Copies and transforms files from the input directory to the output directory
    according to the specified structure and modifications.
    """
    # Define the desired order of classes for 1-based indexing in train.ALL.txt
    # PERS will get index 1, LOC will get index 2, if present.
    # Other classes will follow alphabetically.
    base_ordered_classes = ["PERS", "LOC"]

    # Iterate through each dataset (e.g., CBMA, CDBE, HOME) in the input directory
    for dataset_name in os.listdir(input_dir):
        dataset_path = os.path.join(input_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue # Skip non-directory items

        # Construct the output dataset directory name with optional prefix/suffix
        output_dataset_name = f"{output_prefix}{dataset_name}{output_suffix}"
        output_dataset_path = os.path.join(output_dir, output_dataset_name)
        shutil.rmtree(output_dataset_path, ignore_errors=True)
        os.makedirs(output_dataset_path, exist_ok=True) # Create the output directory

        multiclass_path = os.path.join(dataset_path, "MULTICLASS")
        if not os.path.isdir(multiclass_path):
            print(f"Warning: 'MULTICLASS' directory not found in '{dataset_path}'. Skipping dataset '{dataset_name}'.")
            continue

        # Determine the actual classes present in the current dataset and create a 1-based index map
        current_dataset_single_classes = []
        for class_dir in os.listdir(dataset_path):
            # Collect only directories that are not 'MULTICLASS'
            if os.path.isdir(os.path.join(dataset_path, class_dir)) and class_dir != "MULTICLASS":
                current_dataset_single_classes.append(class_dir)
        
        # Sort current_dataset_single_classes prioritizing PERS, then LOC, then others alphabetically
        sorted_current_dataset_classes_for_indexing = []
        for predefined_class in base_ordered_classes:
            if predefined_class in current_dataset_single_classes:
                sorted_current_dataset_classes_for_indexing.append(predefined_class)
        # Add any other classes found (not PERS or LOC), sorted alphabetically
        for other_class in sorted(c for c in current_dataset_single_classes if c not in base_ordered_classes):
            sorted_current_dataset_classes_for_indexing.append(other_class)

        # Create the 1-based index mapping for train.ALL.txt
        class_index_map = {cls: i + 1 for i, cls in enumerate(sorted_current_dataset_classes_for_indexing)}

        # --- Process and copy files ---

        # 1. Copy priors.json from MULTICLASS
        priors_source_path = os.path.join(multiclass_path, "priors.json")
        priors_dest_path = os.path.join(output_dataset_path, "priors.json")
        if os.path.exists(priors_source_path):
            shutil.copy2(priors_source_path, priors_dest_path)
        else:
            print(f"Warning: 'priors.json' not found in '{multiclass_path}'.")

        # 2. Process test.txt, train.txt, valid.txt (from MULTICLASS, third col = 0)
        # Note: val.txt is renamed to valid.txt
        multiclass_zero_col_files = {
            "test.txt": "test.txt",
            "train.txt": "train.txt",
            "val.txt": "valid.txt"
        }

        for src_file, dest_file in multiclass_zero_col_files.items():
            source_path = os.path.join(multiclass_path, src_file)
            dest_path = os.path.join(output_dataset_path, dest_file)
            if os.path.exists(source_path):
                # Use process_train_file with default None for class_index_map and current_class_name
                # to ensure the third column is always '0' for these files.
                process_train_file(source_path, dest_path)
            else:
                print(f"Warning: '{src_file}' not found in '{multiclass_path}'.")

        # 3. Process train.ALL.txt (from MULTICLASS, third col = 1-based class index)
        train_all_source_path = os.path.join(multiclass_path, "train.txt")
        train_all_dest_path = os.path.join(output_dataset_path, "train.ALL.txt")
        train_entity_dest_path = os.path.join(output_dataset_path, "train.Entity.txt")
        if os.path.exists(train_all_source_path):
            process_train_file(train_all_source_path, train_all_dest_path, class_index_map=class_index_map)
            process_train_file(train_all_source_path, train_entity_dest_path, class_index_map={k:1 for k in class_index_map.keys()})
        else:
            print(f"Warning: 'train.txt' not found in '{multiclass_path}' for 'train.ALL.txt' generation.")

        # 4. Process train.CLASS.txt files (e.g., train.PERS.txt, train.LOC.txt)
        for class_name in sorted_current_dataset_classes_for_indexing:
            class_sub_dir_path = os.path.join(dataset_path, class_name)
            train_class_source_path = os.path.join(class_sub_dir_path, "train.txt")
            train_class_dest_filename = f"train.{class_name}.txt"
            train_class_dest_path = os.path.join(output_dataset_path, train_class_dest_filename)

            if os.path.exists(train_class_source_path):
                process_train_file(train_class_source_path, train_class_dest_path, current_class_name=class_name)
            else:
                print(f"Warning: 'train.txt' not found in '{class_sub_dir_path}' for '{train_class_dest_filename}' generation.")


def main():
    parser = argparse.ArgumentParser(
        description="Copy and transform files from a source directory to a new output directory structure."
    )
    parser.add_argument(
        "--input-dir", 
        required=True, 
        help="Path to the source directory containing datasets (e.g., CBMA, CDBE)."
    )
    parser.add_argument(
        "--output-dir", 
        required=True, 
        help="Path to the destination directory where the transformed data will be saved."
    )
    parser.add_argument(
        "--output-prefix", 
        default="hdsner-", 
        help="Optional prefix string to add to output dataset directory names."
    )
    parser.add_argument(
        "--output-suffix", 
        default="", 
        help="Optional suffix string to add to output dataset directory names (e.g., '_Dict_0.1')."
    )

    args = parser.parse_args()

    # Ensure the main output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    copy_and_transform_data(args.input_dir, args.output_dir, args.output_prefix, args.output_suffix)

if __name__ == "__main__":
    main()
