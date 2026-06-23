from rfdetr import RFDETRLarge

def main():
    model = RFDETRLarge()
    
    # Use raw strings (r"...") for Windows paths to avoid escape character issues
    model.train(
        dataset_dir=r"D:\!staging\@Mestrado\PCBA-AOI\output_sliced",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=r"D:\!staging\@Mestrado\PCBA-AOI\rf-detr_training\output",
    )

if __name__ == '__main__':
    # This block ensures the code only runs when the script is executed directly,
    # not when it is imported by multiprocessing child workers.
    main()