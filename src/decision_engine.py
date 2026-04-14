"""
Decision Engine for Loan Approval
Converts PD/LGD/EAD predictions into actionable approval decisions and pricing
"""
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
from config import FIGURES_DIR


class LoanDecisionEngine:
    """
    Risk-based loan approval and pricing engine.
    
    Implements:
    - Expected Loss (EL) calculation
    - Approval/Rejection decision
    - Risk-based pricing (interest rate adjustment)
    - Portfolio optimization metrics
    """
    
    def __init__(self, approval_el_threshold=0.05, base_rate=0.08, 
                 pricing_scalar=10.0):
        """
        Initialize decision engine.
        
        Args:
            approval_el_threshold: EL threshold for approval (default 5%)
            base_rate: Base interest rate (8%)
            pricing_scalar: Rate adjustment per 1% of EL
        """
        self.approval_el_threshold = approval_el_threshold
        self.base_rate = base_rate
        self.pricing_scalar = pricing_scalar
        self.decisions = None
        self.pricing = None
    
    def calculate_expected_loss(self, pd, lgd, ead):
        """
        Calculate Expected Loss for loans.
        
        EL = PD × LGD × EAD
        
        Args:
            pd: Probability of Default array
            lgd: Loss Given Default array
            ead: Exposure at Default array
        
        Returns:
            expected_loss: Array of EL values
        """
        expected_loss = pd * lgd * ead
        return expected_loss
    
    def make_decisions(self, pd, lgd, ead, loan_amounts=None):
        """
        Generate approval decisions based on expected loss.
        
        Args:
            pd: Probability of Default array
            lgd: Loss Given Default array
            ead: Exposure at Default array
            loan_amounts: Loan amounts (for context)
        
        Returns:
            decision_df: DataFrame with decisions and metrics
        """
        el = self.calculate_expected_loss(pd, lgd, ead)
        
        # Approval/Rejection
        approvals = (el <= self.approval_el_threshold).astype(int)
        
        # Calculate expected loss in currency
        if loan_amounts is not None:
            el_currency = el * loan_amounts
        else:
            el_currency = el
        
        decision_df = pandas.DataFrame({
            'PD': pd,
            'LGD': lgd,
            'EAD': ead,
            'Expected_Loss': el,
            'Expected_Loss_Currency': el_currency,
            'Decision': ['APPROVE' if a == 1 else 'REJECT' for a in approvals],
            'Risk_Level': self._categorize_risk(pd, lgd, el)
        })
        
        self.decisions = decision_df
        return decision_df
    
    def _categorize_risk(self, pd, lgd, el):
        """Categorize loans by risk level."""
        el_pct = el * 100
        
        risk_level = []
        for e in el_pct:
            if e < 1.0:
                risk_level.append('Very Low')
            elif e < 2.5:
                risk_level.append('Low')
            elif e < 5.0:
                risk_level.append('Medium')
            elif e < 10.0:
                risk_level.append('High')
            else:
                risk_level.append('Very High')
        
        return risk_level
    
    def calculate_risk_based_pricing(self, pd, lgd, ead, loan_amounts=None,
                                     min_rate=0.04, max_rate=0.25):
        """
        Calculate risk-based pricing (interest rate adjustment).
        
        Formula:
            Adjusted_Rate = Base_Rate + (EL% × Pricing_Scalar) 
                          + Spread_Buffer
        
        Args:
            pd: Probability of Default array
            lgd: Loss Given Default array
            ead: Exposure at Default array
            loan_amounts: Loan amounts
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
        
        Returns:
            pricing_df: DataFrame with original and adjusted rates
        """
        el = self.calculate_expected_loss(pd, lgd, ead)
        el_pct = el * 100
        
        # Risk-based spread: EL% × pricing_scalar
        risk_spread = el_pct * self.pricing_scalar / 100
        
        # Buffer for operational costs (~0.5%)
        operational_spread = 0.005
        
        # Total spread
        total_spread = risk_spread + operational_spread
        
        # Adjusted rate
        adjusted_rates = self.base_rate + total_spread
        
        # Clip to min/max
        adjusted_rates = np.clip(adjusted_rates, min_rate, max_rate)
        
        # Revenue implications
        if loan_amounts is not None:
            annual_revenue = adjusted_rates * loan_amounts
        else:
            annual_revenue = None
        
        pricing_df = pandas.DataFrame({
            'Base_Rate': self.base_rate,
            'Risk_Spread': risk_spread,
            'Operational_Spread': operational_spread,
            'Adjusted_Rate': adjusted_rates,
            'Annual_Revenue': annual_revenue if annual_revenue is not None else 0,
            'Expected_Loss_Pct': el_pct
        })
        
        self.pricing = pricing_df
        return pricing_df
    
    def get_portfolio_metrics(self):
        """
        Calculate portfolio-level metrics.
        
        Returns:
            metrics_dict: Dictionary with portfolio statistics
        """
        if self.decisions is None:
            raise ValueError("Must call make_decisions() first")
        
        df = self.decisions
        
        metrics = {
            'total_loans': len(df),
            'approved_loans': (df['Decision'] == 'APPROVE').sum(),
            'rejected_loans': (df['Decision'] == 'REJECT').sum(),
            'approval_rate': (df['Decision'] == 'APPROVE').sum() / len(df),
            'avg_pd_approved': df[df['Decision'] == 'APPROVE']['PD'].mean(),
            'avg_pd_rejected': df[df['Decision'] == 'REJECT']['PD'].mean(),
            'avg_el': df['Expected_Loss'].mean(),
            'max_el': df['Expected_Loss'].max(),
            'portfolio_el': df['Expected_Loss_Currency'].sum(),
        }
        
        return metrics
    
    def plot_decision_dashboard(self, save_path=None):
        """
        Plot comprehensive decision dashboard (4 subplots).
        
        Args:
            save_path: Path to save figure
        """
        if self.decisions is None or self.pricing is None:
            raise ValueError("Must call make_decisions() and calculate_risk_based_pricing() first")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Approval vs Rejection
        ax = axes[0, 0]
        decision_counts = self.decisions['Decision'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(decision_counts.index, decision_counts.values, color=colors, edgecolor='black')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Loan Decisions', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Expected Loss Distribution
        ax = axes[0, 1]
        el_approved = self.decisions[self.decisions['Decision'] == 'APPROVE']['Expected_Loss'] * 100
        el_rejected = self.decisions[self.decisions['Decision'] == 'REJECT']['Expected_Loss'] * 100
        ax.hist([el_approved, el_rejected], bins=30, label=['Approved', 'Rejected'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax.axvline(self.approval_el_threshold * 100, color='black', linestyle='--', 
                   linewidth=2, label='Threshold')
        ax.set_xlabel('Expected Loss (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Expected Loss Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Risk-Based Pricing
        ax = axes[1, 0]
        ax.scatter(self.decisions['Expected_Loss'] * 100, 
                   self.pricing['Adjusted_Rate'] * 100,
                   alpha=0.6, s=30, color='#3498db', edgecolor='black')
        ax.set_xlabel('Expected Loss (%)', fontweight='bold')
        ax.set_ylabel('Adjusted Interest Rate (%)', fontweight='bold')
        ax.set_title('Risk-Based Pricing', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. Risk Level Distribution
        ax = axes[1, 1]
        risk_counts = self.decisions['Risk_Level'].value_counts()
        risk_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        risk_counts = risk_counts.reindex([r for r in risk_order if r in risk_counts.index])
        colors_risk = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        ax.bar(range(len(risk_counts)), risk_counts.values, 
               color=colors_risk[:len(risk_counts)], edgecolor='black')
        ax.set_xticks(range(len(risk_counts)))
        ax.set_xticklabels(risk_counts.index, rotation=45)
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Loan Portfolio Risk Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Loan Decision Engine Dashboard', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        plt.close()
    
    def generate_decision_report(self, decisions_df, pricing_df, save_dir=None):
        """
        Generate text report with key metrics.
        
        Args:
            decisions_df: Decisions DataFrame
            pricing_df: Pricing DataFrame
            save_dir: Directory to save report
        
        Returns:
            report_text: Report as string
        """
        metrics = self.get_portfolio_metrics()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              LOAN DECISION ENGINE REPORT                      ║
╚══════════════════════════════════════════════════════════════╝

PORTFOLIO OVERVIEW:
  • Total Loans:           {metrics['total_loans']:,}
  • Approved:              {metrics['approved_loans']:,} ({metrics['approval_rate']:.1%})
  • Rejected:              {metrics['rejected_loans']:,} ({1 - metrics['approval_rate']:.1%})

EXPECTED LOSS METRICS:
  • Average Portfolio EL:  {metrics['avg_el']:.2%}
  • Maximum Loan EL:       {metrics['max_el']:.2%}
  • Total Approved EL:     ${metrics['portfolio_el']:,.2f}

PD ANALYSIS:
  • Avg PD (Approved):     {metrics['avg_pd_approved']:.2%}
  • Avg PD (Rejected):     {metrics['avg_pd_rejected']:.2%}

PRICING SUMMARY:
  • Base Rate:             {self.base_rate:.2%}
  • Avg Adjusted Rate:     {pricing_df['Adjusted_Rate'].mean():.2%}
  • Min Rate:              {pricing_df['Adjusted_Rate'].min():.2%}
  • Max Rate:              {pricing_df['Adjusted_Rate'].max():.2%}
  • Total Annual Revenue:  ${pricing_df['Annual_Revenue'].sum():,.2f}

DECISION THRESHOLDS:
  • EL Approval Threshold: {self.approval_el_threshold:.2%}
  • Pricing Scalar:        {self.pricing_scalar}

═══════════════════════════════════════════════════════════════
"""
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            report_path = os.path.join(save_dir, "decision_engine_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"  Saved report: {report_path}")
        
        return report
