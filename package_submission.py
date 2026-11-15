import os
import zipfile

def package_project(output_zip="DIP_Traffic_Sign_Project_Submission.zip"):
    print(f"Creating full project submission package: {output_zip}...")
    
    include_files = [
        "app.py",
        "run_cli_benchmark.py",
        "generate_samples.py",
        "generate_video.py",
        "generate_report_visuals.py",
        "generate_all_screenshots.py",
        "generate_presentation_pptx.py",
        "benchmark_results.csv",
        "Project_Report.md",
        "Presentation_Guide.md",
        "Presentation_Slides.pptx",
        "README.md"
    ]
    
    include_dirs = [
        "pipeline",
        "sample_data",
        "screenshots",
        "report_assets"
    ]
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add root files
        for f in include_files:
            if os.path.exists(f):
                zipf.write(f, arcname=f)
                print(f"  + Added: {f}")
        
        # Add directories
        for d in include_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for file in files:
                        if not file.endswith('.pyc'):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, ".")
                            zipf.write(file_path, arcname=arcname)
                print(f"  + Added directory: {d}/")

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n[✓] Submission package successfully generated: {output_zip} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    package_project()

