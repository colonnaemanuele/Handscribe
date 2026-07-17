import streamlit as st
import subprocess
import os
import sys
import tempfile
import re

def parse_and_display_results(content: str, results_dir: str):
    """
    Parses the output log from seq_eval and displays metrics.
    """
    st.subheader("📊 Evaluation Results")
    
    # Regex to find the line with test results
    # Example: "Epoch 6667, test Results -> ROUGE-L: 35.1234, BLEU-1: 50.4321, ..."
    match = re.search(r"test Results -> (.*)", content)
    
    if match:
        results_str = match.group(1).strip()
        
        metrics = {}
        try:
            pairs = results_str.split(",")
            if len(pairs) == 0 or ':' not in results_str:
                raise ValueError("No key-value pairs found")

            # Create columns for each metric
            cols = st.columns(len(pairs))
            
            for i, pair in enumerate(pairs):
                key, val = pair.split(":")
                key = key.strip()
                val = float(val.strip())
                metrics[key] = val
                
                # Use Streamlit's metric component
                cols[i].metric(label=key, value=f"{val:.4f}")
            
            st.json(metrics)
            
            # Show where the full output file is
            hyp_path = os.path.join(results_dir, "output-hypothesis-test.ctm")
            st.info(f"Full hypothesis file (CTM format) saved to: {hyp_path}")

        except Exception as e:
            st.error(f"Failed to parse results string: '{results_str}'")
            st.error(f"Error: {e}")
            st.text("Raw log content:")
            st.code(content, language="text")
            
    else:
        st.error("Could not find 'test Results ->' line in the log file.")
        st.text("Raw log content:")
        st.code(content, language="text")

def main():
    st.set_page_config(layout="wide")
    st.title("Sign Language Translation: Full Test Set Evaluation")
    
    st.info(
        "**Prerequisite:** This Streamlit app must be run from the root of your "
        "`handscribe` project directory.\n"
        "Example: `streamlit run demo.py`"
    )

    # --- 1. User Inputs ---
    st.header("1. Upload Artifacts")
    col1, col2 = st.columns(2)
    
    with col1:
        config_file = st.file_uploader(
            "Upload your **base config file**",
            type=["yaml", "yml"],
            help="This is the main .yaml file (e.g., `baseline.yaml`) "
                 "you used to define the experiment."
        )
    
    with col2:
        st.markdown(
            "**Checkpoint file is larger than 4GB?**\n"
            "➡️ Instead of uploading, enter the **absolute path** to the checkpoint file on the server below."
        )
        checkpoint_path_input = st.text_input(
            "Or enter checkpoint path (for files >4GB)",
            value="",
            help="If your checkpoint is too large to upload, enter its absolute path here."
        )
        checkpoint_file = st.file_uploader(
            "Or upload your **model checkpoint** (<4GB)",
            type=["pt"],
            help="The trained model checkpoint (.pt file) you want to evaluate."
        )

    st.header("2. Configure Evaluation")
    dataset = st.selectbox(
        "Select the Dataset",
        ("phoenix2014-T", "CSL", "CSL-Daily", "lis"),
        help="This must match the dataset used for training, "
             "as it's used to load the correct dataset info file."
    )

    # --- 3. Run Evaluation ---
    st.header("3. Run Evaluation")
    
    if st.button("Start Evaluation on Test Set", type="primary"):
        if config_file and (checkpoint_file or checkpoint_path_input) and dataset:
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = os.path.join(tmpdir, "config.yaml")
                results_dir = os.path.join(tmpdir, "results")
                os.makedirs(results_dir, exist_ok=True)

                with open(config_path, "wb") as f:
                    f.write(config_file.getvalue())

                # Determine checkpoint path
                if checkpoint_path_input.strip():
                    checkpoint_path = checkpoint_path_input.strip()
                    if not os.path.exists(checkpoint_path):
                        st.error(f"Checkpoint path does not exist: {checkpoint_path}")
                        return
                else:
                    checkpoint_path = os.path.join(tmpdir, "model.pt")
                    with open(checkpoint_path, "wb") as f:
                        f.write(checkpoint_file.getvalue())
                
                st.info(
                    f"Running evaluation in temporary directory: {tmpdir}\n"
                    f"Results will be saved to: {results_dir}"
                )

                command = [
                    sys.executable,
                    "main.py",
                    "--phase", "eval",
                    "--config", config_path,
                    "--load-checkpoint", checkpoint_path,
                    "--dataset", dataset,
                    "--work-dir", results_dir,
                ]
                
                st.code(" ".join(command), language="bash")

                with st.spinner("⏳ Running evaluation on the entire test set... "
                               "This may take several minutes."):
                    try:
                        result = subprocess.run(
                            command, 
                            capture_output=True, 
                            text=True, 
                            encoding='utf-8', 
                            timeout=7200 
                        )
                        if result.returncode == 0:
                            st.success("✅ Evaluation Finished Successfully!")
                            results_file_path = os.path.join(results_dir, "test.txt")
                            if os.path.exists(results_file_path):
                                with open(results_file_path, "r") as f:
                                    log_content = f.read()
                                parse_and_display_results(log_content, results_dir)
                            else:
                                st.error(f"Could not find results file at: {results_file_path}")
                            with st.expander("Show Full STDOUT Log"):
                                st.code(result.stdout, language="log")
                        else:
                            st.error(f"❌ Evaluation Failed! (Exit Code: {result.returncode})")
                            st.subheader("STDOUT Log:")
                            st.code(result.stdout, language="log")
                            st.subheader("STDERR Log:")
                            st.code(result.stderr, language="log")
                    except subprocess.TimeoutExpired:
                        st.error("❌ Evaluation Timed Out after 2 hours.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        st.exception(e)
        else:
            st.warning("Please upload a config file and a checkpoint (or enter its path), and select a dataset.")

if __name__ == "__main__":
    main()


