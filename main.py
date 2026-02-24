"""
Main entry point for Sign Language Recognition System.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def print_menu():
    """Print main menu."""
    print("\n" + "="*60)
    print("Sign Language Recognition System")
    print("="*60)
    print("\nChoose an option:")
    print("  1. Collect Data (Manual/Auto)")
    print("  2. Train MLP Model (Single Frame)")
    print("  3. Train LSTM Model (Sequence/Temporal)")
    print("  4. Run GUI with MLP (Fast)")
    print("  5. Run GUI with LSTM (Smart/Temporal)")
    print("  6. Check Data Balance")
    print("  7. Exit")
    print("="*60)


def main():
    """Main function."""
    while True:
        print_menu()
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            from collect import main as collect_main
            collect_main()
            
        elif choice == "2":
            from train import train
            train()
            
        elif choice == "3":
            from train_lstm import train_lstm
            train_lstm()
            
        elif choice == "4":
            from gui import main as gui_main
            gui_main()
            
        elif choice == "5":
            from gui_lstm import main as gui_lstm_main
            gui_lstm_main()
            
        elif choice == "6":
            from utils import check_data_balance
            from config import config
            
            print("\n" + "="*60)
            print("DATA BALANCE CHECK")
            print("="*60)
            
            balance = check_data_balance()
            print(f"\nTotal samples: {balance['total']}")
            print(f"Min per class: {balance['min']}")
            print(f"Max per class: {balance['max']}")
            
            print("\nPer-class counts:")
            for label, count in sorted(balance['counts'].items()):
                status = "✓" if count >= config.COLLECTION["min_samples_per_class"] else "⚠"
                print(f"  {status} {label}: {count}")
            
            if balance['warnings']:
                print("\n⚠ Warnings:")
                for w in balance['warnings']:
                    print(f"  - {w}")
            else:
                print("\n✓ Data is well balanced!")
            
            print("="*60)
            
        elif choice == "7":
            print("\nGoodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1-7.")


if __name__ == "__main__":
    main()
