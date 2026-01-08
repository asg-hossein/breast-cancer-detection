"""
Main breast cancer classification pipeline with interactive menu
Replaces all pipelines
"""
import sys
import os
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from visualizer import DataVisualizer, ModelVisualizer
from fuzzy_enhancer import FuzzyEnhancer
from utils import ensure_directories


def display_interactive_menu():
    print("\n" + "="*60)
    print("Breast Cancer Classification Pipeline")
    print("="*60)
    
    print("\nPlease select execution mode:")
    print("1️ Basic        - Basic data processing")
    print("2️ Preprocessed - With full preprocessing (PCA, outlier removal)")
    print("3️ Fuzzy        - With fuzzy enhancement and clustering")
    print("4️ Full         - Complete analysis (all stages)")
    print("5️ All          - Run all modes sequentially")
    print("0️ Exit         - Exit program")
    print("="*60)
    
    while True:
        try:
            choice = input("\nEnter desired option number (0-5): ").strip()
            
            if choice == '0':
                print("Exiting program...")
                sys.exit(0)
            
            mode_mapping = {
                '1': 'basic',
                '2': 'preprocessed',
                '3': 'fuzzy',
                '4': 'full',
                '5': 'all'
            }
            
            if choice in mode_mapping:
                selected_mode = mode_mapping[choice]
                
                use_pca = False
                remove_outliers = True
                output_prefix = ''
                
                if selected_mode in ['preprocessed', 'fuzzy', 'all']:
                    pca_input = input("Use PCA for dimensionality reduction? (y/n): ").strip().lower()
                    use_pca = pca_input in ['y', 'yes']
                
                if selected_mode != 'basic':
                    outliers_input = input("Remove outliers? (y/n) [default: y]: ").strip().lower()
                    remove_outliers = outliers_input not in ['n', 'no']
                
                prefix_input = input("🔹 Output file name prefix (optional, press Enter to skip): ").strip()
                if prefix_input:
                    output_prefix = prefix_input
                
                return selected_mode, use_pca, remove_outliers, output_prefix
            else:
                print("Invalid selection! Please enter a number 0-5.")
                
        except KeyboardInterrupt:
            print("\n\nExiting program...")
            sys.exit(0)
        except Exception as e:
            print(f"Input error: {e}")


def parse_arguments():
    if len(sys.argv) == 1:
        mode, use_pca, remove_outliers, output_prefix = display_interactive_menu()
        
        sys.argv.append(mode)
        if use_pca:
            sys.argv.append('--use_pca')
        if remove_outliers:
            sys.argv.append('--remove_outliers')
        if output_prefix:
            sys.argv.append('--output_prefix')
            sys.argv.append(output_prefix)
    
    parser = argparse.ArgumentParser(
        description='Breast Cancer Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python main_pipeline.py                      # Display interactive menu
  python main_pipeline.py basic                # Run basic pipeline
  python main_pipeline.py preprocessed         # With full preprocessing
  python main_pipeline.py fuzzy --use_pca      # With fuzzy enhancement and PCA
  python main_pipeline.py full --remove_outliers  # Complete analysis with outlier removal
  python main_pipeline.py all                  # Run all modes
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['basic', 'preprocessed', 'fuzzy', 'full', 'all'],
        nargs='?',
        help='Pipeline execution mode'
    )
    
    parser.add_argument(
        '--use_pca',
        action='store_true',
        help='Use PCA (only in preprocessed and fuzzy modes)'
    )
    
    parser.add_argument(
        '--remove_outliers',
        action='store_true',
        default=True,
        help='Remove outliers (default: True)'
    )
    
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='',
        help='Output file name prefix'
    )
    
    return parser.parse_args()


def run_basic_pipeline(output_prefix=''):
    print("\n" + "="*60)
    print("Running basic pipeline...")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{output_prefix}basic_{timestamp}" if output_prefix else f"basic_{timestamp}"
    
    processor = DataProcessor()
    X_train, X_test, y_train, y_test, _ = processor.get_processed_data(
        use_pca=False, 
        remove_outliers=False
    )
    
    trainer = ModelTrainer()
    trainer.initialize_models()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    evaluator = ModelEvaluator(results)
    evaluator.create_summary_dataframe()
    evaluator.save_results_to_csv(f"{prefix}_results.csv")
    evaluator.print_comparison_table()
    
    visualizer = ModelVisualizer(results)
    visualizer.plot_all_model_charts()
    
    print(f"\nBasic pipeline executed successfully")
    print(f"Results with prefix: {prefix}")
    
    return results


def run_preprocessed_pipeline(use_pca=False, remove_outliers=True, output_prefix=''):
    print("\n" + "="*60)
    print(f"Running preprocessed pipeline (PCA: {use_pca})...")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pca_str = "with_pca" if use_pca else "no_pca"
    prefix = f"{output_prefix}preprocessed_{pca_str}_{timestamp}" if output_prefix else f"preprocessed_{pca_str}_{timestamp}"
    
    processor = DataProcessor()
    X_train, X_test, y_train, y_test, _ = processor.get_processed_data(
        use_pca=use_pca, 
        remove_outliers=remove_outliers
    )
    
    trainer = ModelTrainer()
    trainer.initialize_models()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    evaluator = ModelEvaluator(results)
    evaluator.create_summary_dataframe()
    evaluator.save_results_to_csv(f"{prefix}_results.csv")
    evaluator.save_detailed_results(f"{prefix}_detailed.json")
    evaluator.print_comparison_table()
    
    visualizer = ModelVisualizer(results)
    visualizer.plot_all_model_charts()
    
    print(f"\nPreprocessed pipeline executed successfully")
    print(f"Results with prefix: {prefix}")
    
    return results


def run_fuzzy_pipeline(use_pca=False, output_prefix=''):
    print("\n" + "="*60)
    print(f"Running fuzzy enhancement pipeline (PCA: {use_pca})...")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pca_str = "with_pca" if use_pca else "no_pca"
    prefix = f"{output_prefix}fuzzy_{pca_str}_{timestamp}" if output_prefix else f"fuzzy_{pca_str}_{timestamp}"
    
    processor = DataProcessor()
    X_train, X_test, y_train, y_test, _ = processor.get_processed_data(
        use_pca=use_pca, 
        remove_outliers=True
    )
    
    trainer = ModelTrainer()
    trainer.initialize_models()
    base_results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    best_model, best_model_name = trainer.get_best_model()
    base_accuracy = base_results[best_model_name]["metrics"]["accuracy"]
    
    print(f"\nBest model for fuzzy enhancement: {best_model_name}")
    print(f"   Base accuracy: {base_accuracy:.4f}")
    
    fuzzy_enhancer = FuzzyEnhancer()
    enhanced_model, enhanced_metrics = fuzzy_enhancer.enhance_with_fuzzy_features(
        X_train, X_test, y_train, y_test, best_model
    )
    
    improvement = fuzzy_enhancer.compare_with_base_model(
        best_model_name, base_accuracy, 
        enhanced_metrics["accuracy"], enhanced_metrics
    )
    
    fuzzy_enhancer.save_enhanced_model(enhanced_model, f"{prefix}_enhanced")
    
    results = {
        "Base Model": {best_model_name: base_results[best_model_name]["metrics"]},
        "Fuzzy Enhanced": {f"{best_model_name}_Fuzzy": enhanced_metrics}
    }
    
    evaluator = ModelEvaluator(results)
    evaluator.save_results_to_csv(f"{prefix}_comparison.csv")
    
    quality_report = fuzzy_enhancer.get_fuzzy_quality_report()
    print(f"\nFuzzy quality report:")
    for key, value in quality_report.items():
        print(f"   {key}: {value}")
    
    print(f"\nFuzzy enhancement pipeline executed successfully")
    print(f"Results with prefix: {prefix}")
    
    return {
        "base_results": base_results,
        "enhanced_model": enhanced_model,
        "improvement": improvement
    }


def run_full_analysis_pipeline(output_prefix=''):
    print("\n" + "="*60)
    print("Running complete analysis...")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{output_prefix}full_analysis_{timestamp}" if output_prefix else f"full_analysis_{timestamp}"
    
    processor = DataProcessor()
    processor.load_data()
    processor.handle_missing_values()
    
    data_viz = DataVisualizer(processor)
    
    if processor.data is not None:
        features_data = processor.data.drop('diagnosis', axis=1)
        data_viz.plot_correlation_heatmap(features_data)
    
    X_train, X_test, y_train, y_test, scaler = processor.get_processed_data(
        use_pca=False, 
        remove_outliers=False
    )
    
    data_viz.plot_class_distribution(processor.y)
    
    feature_names = processor.X.columns.tolist()
    data_viz.plot_feature_histograms(X_train, feature_names)
    
    trainer = ModelTrainer()
    trainer.initialize_models()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    evaluator = ModelEvaluator(results)
    evaluator.create_summary_dataframe()
    evaluator.save_results_to_csv(f"{prefix}_results.csv")
    evaluator.save_detailed_results(f"{prefix}_detailed.json")
    evaluator.print_comparison_table()
    
    model_viz = ModelVisualizer(results)
    model_viz.plot_all_model_charts()
    
    best_info = evaluator.get_best_model_info()
    if best_info:
        print(f"\nOverall best model: {best_info['model_name']}")
        print(f"   Accuracy: {best_info['accuracy']}")
    
    print(f"\nComplete analysis executed successfully")
    print(f"Results with prefix: {prefix}")
    
    return results


def main():
    args = parse_arguments()
    
    print("="*60)
    print("Breast Cancer Classification Pipeline")
    print("="*60)
    print(f"Selected mode: {args.mode}")
    print(f" Settings: PCA={args.use_pca}, Remove Outliers={args.remove_outliers}")
    if args.output_prefix:
        print(f"Output prefix: {args.output_prefix}")
    print("="*60)
    
    ensure_directories()
    
    all_results = {}
    
    if args.mode == 'basic' or args.mode == 'all':
        all_results['basic'] = run_basic_pipeline(args.output_prefix)
    
    if args.mode == 'preprocessed' or args.mode == 'all':
        all_results['preprocessed'] = run_preprocessed_pipeline(
            use_pca=args.use_pca,
            remove_outliers=args.remove_outliers,
            output_prefix=args.output_prefix
        )
    
    if args.mode == 'fuzzy' or args.mode == 'all':
        all_results['fuzzy'] = run_fuzzy_pipeline(
            use_pca=args.use_pca,
            output_prefix=args.output_prefix
        )
    
    if args.mode == 'full' or args.mode == 'all':
        all_results['full'] = run_full_analysis_pipeline(args.output_prefix)
    
    print("\n" + "="*60)
    print("Pipeline execution completed!")
    print("="*60)
    
    if args.mode == 'all':
        print("\nSummary of all mode executions:")
        print("   Basic pipeline")
        print("   Preprocessed pipeline")
        print("   Fuzzy enhancement pipeline")
        print("   Complete analysis")
        print(f"\nAll results saved in results/ folder")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    print("\nFor reuse, choose one of the following methods:")
    print("   • python main_pipeline.py                    (interactive menu)")
    print("   • python main_pipeline.py [mode] --help      (command line help)")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program execution stopped by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please ensure all required libraries are installed:")
        print("pip install scikit-learn pandas numpy matplotlib seaborn scikit-fuzzy joblib")
        sys.exit(1)