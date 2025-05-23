"""
Εργαλεία οπτικοποίησης για ψηφιακό δίδυμο διαβήτη.

Περιλαμβάνει:
- Γραφήματα προβλέψεων και γλυκόζης
- Clarke Error Grid plots
- Training history plots
- Interactive dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Ρυθμίσεις matplotlib για ελληνικά
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def plot_predictions(y_true: np.ndarray, 
                    y_pred: np.ndarray,
                    timestamps: Optional[np.ndarray] = None,
                    title: str = "Προβλέψεις Γλυκόζης",
                    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Γράφημα σύγκρισης πραγματικών τιμών και προβλέψεων.
    
    Args:
        y_true: Πραγματικές τιμές
        y_pred: Προβλέψεις
        timestamps: Χρονοσφραγίδες (προαιρετικό)
        title: Τίτλος γραφήματος
        confidence_intervals: (lower, upper) για confidence intervals
        figsize: Μέγεθος γραφήματος
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if timestamps is None:
        x = np.arange(len(y_true))
        xlabel = "Δείγματα"
    else:
        x = timestamps
        xlabel = "Χρόνος"
    
    # Πραγματικές τιμές
    ax.plot(x, y_true, 'b-', label='Πραγματικές τιμές', linewidth=2, alpha=0.8)
    
    # Προβλέψεις
    ax.plot(x, y_pred, 'r--', label='Προβλέψεις', linewidth=2, alpha=0.8)
    
    # Confidence intervals
    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        ax.fill_between(x, lower, upper, alpha=0.3, color='red', 
                       label='Διάστημα εμπιστοσύνης')
    
    # Ζώνες γλυκόζης
    ax.axhspan(70, 180, alpha=0.1, color='green', label='Στόχος (70-180 mg/dL)')
    ax.axhspan(0, 70, alpha=0.1, color='orange', label='Χαμηλή (<70 mg/dL)')
    ax.axhspan(180, 400, alpha=0.1, color='red', label='Υψηλή (>180 mg/dL)')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Γλυκόζη (mg/dL)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_glucose_trend(glucose_data: pd.Series,
                      meals: Optional[pd.Series] = None,
                      insulin: Optional[pd.Series] = None,
                      title: str = "Τάση Γλυκόζης",
                      figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """
    Γράφημα τάσης γλυκόζης με γεύματα και ινσουλίνη.
    
    Args:
        glucose_data: Δεδομένα γλυκόζης
        meals: Γεύματα (προαιρετικό)
        insulin: Ινσουλίνη (προαιρετικό)
        title: Τίτλος γραφήματος
        figsize: Μέγεθος γραφήματος
        
    Returns:
        Matplotlib Figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Γράφημα γλυκόζης
    ax1.plot(glucose_data.index, glucose_data.values, 'b-', linewidth=2)
    
    # Ζώνες γλυκόζης
    ax1.axhspan(70, 180, alpha=0.1, color='green', label='Στόχος')
    ax1.axhspan(0, 70, alpha=0.1, color='orange', label='Χαμηλή')
    ax1.axhspan(180, 400, alpha=0.1, color='red', label='Υψηλή')
    
    ax1.set_ylabel('Γλυκόζη (mg/dL)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Γράφημα γευμάτων
    if meals is not None:
        ax2.stem(meals.index, meals.values, basefmt=' ', linefmt='g-', markerfmt='go')
        ax2.set_ylabel('Υδατάνθρακες (g)')
        ax2.set_title('Γεύματα')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_visible(False)
    
    # Γράφημα ινσουλίνης
    if insulin is not None:
        ax3.stem(insulin.index, insulin.values, basefmt=' ', linefmt='r-', markerfmt='ro')
        ax3.set_ylabel('Ινσουλίνη (U)')
        ax3.set_title('Ινσουλίνη')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.set_visible(False)
    
    ax3.set_xlabel('Χρόνος')
    plt.tight_layout()
    return fig


def plot_clarke_grid(y_true: np.ndarray, 
                    y_pred: np.ndarray,
                    title: str = "Clarke Error Grid",
                    figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Clarke Error Grid plot.
    
    Args:
        y_true: Πραγματικές τιμές
        y_pred: Προβλέψεις
        title: Τίτλος γραφήματος
        figsize: Μέγεθος γραφήματος
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Σκατεροδιάγραμμα
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Γραμμή y=x (τέλεια ακρίβεια)
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([0, max_val], [0, max_val], 'k-', linewidth=2, alpha=0.7, label='Τέλεια ακρίβεια')
    
    # Ζώνες Clarke Grid (απλουστευμένες)
    ax.fill_between([0, 70], [0, 70], [0, 180], alpha=0.1, color='green', label='Zone A')
    ax.fill_between([70, 180], [56, 144], [84, 216], alpha=0.1, color='yellow', label='Zone B')
    
    ax.set_xlabel('Πραγματικές τιμές (mg/dL)')
    ax.set_ylabel('Προβλέψεις (mg/dL)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    plt.tight_layout()
    return fig


def plot_training_history(history: Dict,
                         title: str = "Ιστορικό Εκπαίδευσης",
                         figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Γράφημα ιστορικού εκπαίδευσης.
    
    Args:
        history: Dictionary με train/val losses
        title: Τίτλος γραφήματος
        figsize: Μέγεθος γραφήματος
        
    Returns:
        Matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss κατά την εκπαίδευση')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE (αν υπάρχει)
    if 'train_rmse' in history:
        ax2.plot(epochs, history['train_rmse'], 'b-', label='Training RMSE', linewidth=2)
        ax2.plot(epochs, history['val_rmse'], 'r-', label='Validation RMSE', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE κατά την εκπαίδευση')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')
    
    plt.tight_layout()
    return fig


def plot_time_in_range(tir_results: Dict,
                      title: str = "Time in Range Ανάλυση",
                      figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Γράφημα Time in Range.
    
    Args:
        tir_results: Αποτελέσματα TIR από utils.metrics
        title: Τίτλος γραφήματος
        figsize: Μέγεθος γραφήματος
        
    Returns:
        Matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Pie chart
    categories = ['Πολύ Χαμηλή\n(<54)', 'Χαμηλή\n(54-70)', 
                 'Στόχος\n(70-180)', 'Υψηλή\n(180-250)', 'Πολύ Υψηλή\n(>250)']
    values = [tir_results['very_low_percent'], tir_results['low_percent'],
             tir_results['in_range_percent'], tir_results['high_percent'],
             tir_results['very_high_percent']]
    colors = ['darkred', 'orange', 'green', 'red', 'darkred']
    
    ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Κατανομή Γλυκόζης')
    
    # Bar chart
    ax2.barh(categories, values, color=colors, alpha=0.7)
    ax2.set_xlabel('Ποσοστό χρόνου (%)')
    ax2.set_title('Time in Range')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_dict: Dict,
                           title: str = "Σύγκριση Μετρικών",
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Σύγκριση μετρικών διαφορετικών μοντέλων.
    
    Args:
        metrics_dict: {model_name: metrics_results}
        title: Τίτλος γραφήματος
        figsize: Μέγεθος γραφήματος
        
    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    model_names = list(metrics_dict.keys())
    
    # RMSE
    rmse_values = [metrics_dict[model]['rmse'] for model in model_names]
    axes[0].bar(model_names, rmse_values, color='skyblue', alpha=0.7)
    axes[0].set_title('RMSE')
    axes[0].set_ylabel('mg/dL')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MAE
    mae_values = [metrics_dict[model]['mae'] for model in model_names]
    axes[1].bar(model_names, mae_values, color='lightgreen', alpha=0.7)
    axes[1].set_title('MAE')
    axes[1].set_ylabel('mg/dL')
    axes[1].tick_params(axis='x', rotation=45)
    
    # MAPE
    mape_values = [metrics_dict[model]['mape'] for model in model_names]
    axes[2].bar(model_names, mape_values, color='lightcoral', alpha=0.7)
    axes[2].set_title('MAPE')
    axes[2].set_ylabel('%')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Clarke Zone A
    if 'clarke_grid' in metrics_dict[model_names[0]]:
        zone_a_values = [metrics_dict[model]['clarke_grid']['A'] for model in model_names]
        axes[3].bar(model_names, zone_a_values, color='gold', alpha=0.7)
        axes[3].set_title('Clarke Zone A')
        axes[3].set_ylabel('%')
        axes[3].tick_params(axis='x', rotation=45)
    else:
        axes[3].axis('off')
    
    plt.tight_layout()
    return fig


def create_dashboard(twin_model, 
                    test_data: pd.DataFrame,
                    predictions: np.ndarray,
                    title: str = "Dashboard Ψηφιακού Διδύμου") -> plt.Figure:
    """
    Δημιουργία comprehensive dashboard.
    
    Args:
        twin_model: Το εκπαιδευμένο μοντέλο
        test_data: Test δεδομένα
        predictions: Προβλέψεις
        title: Τίτλος dashboard
        
    Returns:
        Matplotlib Figure
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Κεντρικός τίτλος
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
    
    # Κεντρικό γράφημα προβλέψεων
    ax_main = fig.add_subplot(gs[0, :2])
    if 'cgm' in test_data.columns:
        y_true = test_data['cgm'].values
        plot_predictions(y_true, predictions, title="Προβλέψεις vs Πραγματικές Τιμές")
    
    # Clarke Grid
    ax_clarke = fig.add_subplot(gs[0, 2])
    if 'cgm' in test_data.columns:
        plot_clarke_grid(y_true, predictions, title="Clarke Grid")
    
    # Μετρικές
    ax_metrics = fig.add_subplot(gs[1, 0])
    if hasattr(twin_model, 'get_training_history'):
        history = twin_model.get_training_history()
        if history:
            epochs = range(1, len(history['train_losses']) + 1)
            ax_metrics.plot(epochs, history['train_losses'], 'b-', label='Train')
            ax_metrics.plot(epochs, history['val_losses'], 'r-', label='Val')
            ax_metrics.set_title('Training History')
            ax_metrics.legend()
    
    # Model info
    ax_info = fig.add_subplot(gs[1, 1:])
    ax_info.axis('off')
    model_info = f"Μοντέλο: {twin_model.__class__.__name__}\n"
    if hasattr(twin_model, 'get_params'):
        params = twin_model.get_params()
        for key, value in list(params.items())[:5]:  # Πρώτες 5 παραμέτρους
            model_info += f"{key}: {value}\n"
    ax_info.text(0.1, 0.9, model_info, transform=ax_info.transAxes, 
                fontsize=10, verticalalignment='top')
    
    # Στατιστικά
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    if 'cgm' in test_data.columns:
        from .metrics import calculate_metrics
        metrics = calculate_metrics(y_true, predictions)
        stats_text = f"RMSE: {metrics['rmse']:.2f} mg/dL | "
        stats_text += f"MAE: {metrics['mae']:.2f} mg/dL | "
        stats_text += f"MAPE: {metrics['mape']:.2f}%"
        ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                     fontsize=12, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    return fig 