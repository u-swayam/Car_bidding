# Core formula (evaluated every round):
#   bankroll_ratio  = current_bankroll / INITIAL_BANKROLL
#   margin          = BASE_MARGIN × (bankroll_ratio ^ 0.5)
#   max_bid         = predicted_value × (1 − margin)
#                     capped at MAX_SINGLE_SPEND_PCT × bankroll
#
#   increment       = max(MIN_INCREMENT, current_highest_bid × INCREMENT_PCT)
#   next_bid        = current_highest_bid + increment

import os
import pickle
import numpy as np


# Constants 
INITIAL_BANKROLL     = 500_000
BASE_MARGIN          = 0.12    # target profit margin per car (12%)
BANKROLL_EXP         = 0.5     # sqrt dampening on bankroll effect
MIN_PROFIT_PCT       = 0.04    # skip car if remaining margin < 4%
MIN_INCREMENT        = 200     # minimum bid increment ($)
INCREMENT_PCT        = 0.01    # increment = max(MIN_INCREMENT, bid × 1%)
MAX_SINGLE_SPEND_PCT = 0.20    # never spend > 20% of bankroll on one car
BANKROLL_FLOOR_PCT   = 0.05    # stop bidding entirely below 5% bankroll
CURRENT_YEAR         = 2026
CAT_COLS             = ['make', 'model', 'trim', 'body', 'transmission',
                        'state', 'color', 'interior']


round
class BiddingAgent:

    def __init__(self):
        # Load files with relative paths (arena requirement)
        base = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(base, 'model_SwayamUpadhyaya.pkl'), 'rb') as f:
            payload = pickle.load(f)

        # Unpack model payload
        self.model            = payload['model']
        self.feature_cols     = payload['feature_cols']

        # Cleaning stats — computed from train only
        self.odo_cap          = payload['odo_cap']
        self.odo_median_train = payload['odo_median_train']
        self.year_cond_map    = payload['year_cond_map']   # {year: median_condition}
        self.global_cond      = payload['global_cond']     # fallback for unseen years
        self.rare_map         = payload['rare_map']        # {col: set of rare values}

        # Target encoding maps — full-train smoothed means
        self.te_means         = payload['te_means']        # {col: {cat: smoothed_mean}}
        self.te_global_mean   = payload['te_global_mean']  # fallback for unseen categories

        # Session state (no upper bound on cars_seen)
        self.bankroll         = INITIAL_BANKROLL
        self.cars_seen        = 0    
        self.cars_won         = 0
        self.total_spent      = 0.0
        self.total_profit     = 0.0

        # Per-car state (reset in analyze_item every car) 
        self.predicted_value  = None
        self.max_bid          = None


    def _clean_year(self, raw) -> float:
        try:
            val = float(raw)
            if 1982 <= val <= CURRENT_YEAR:
                return val
        except (TypeError, ValueError):
            pass
        return float(CURRENT_YEAR - 14)
        #default to 14-year-old car (if missing/invalid) as median age in train was 14 
    def _clean_odometer(self, raw) -> float:
        """Coerce -> cap at train odo_cap -> fill missing with train median."""
        try:
            val = float(raw)
            if not np.isnan(val):
                return min(val, self.odo_cap)
        except (TypeError, ValueError):
            pass
        return self.odo_median_train

    def _impute_condition(self, raw, year: float) -> float:
        """
        Mirror train imputation:
          present & valid (1-5) -> use as-is
          missing/invalid       -> per-year median from train
          year unseen in train  -> global train median
        """
        try:
            val = float(raw)
            if 1.0 <= val <= 5.0:
                return val
        except (TypeError, ValueError):
            pass
        return self.year_cond_map.get(year, self.global_cond)

    def _clean_cat(self, col: str, raw) -> str:
        """
        Mirror train categorical cleaning:
          fill None -> 'Unknown'
          title-case + strip
          rare values (from train rare_map) -> 'Other'
        """
        if raw is None or str(raw).strip().lower() in ('nan', 'none', ''):
            val = 'Unknown'
        else:
            val = str(raw).strip().title()

        if col in self.rare_map and val in self.rare_map[col]:
            return 'Other'
        return val


    def _target_encode(self, col: str, val: str) -> float:
        """
        Look up the smoothed target-encoded mean for a category.
        Uses full-train means (same as what val set used in model.ipynb).
        Unseen category -> global_mean fallback.
        """
        return self.te_means[col].get(val, self.te_global_mean)


    def _preprocess(self, car: dict) -> np.ndarray:
        """
        Full single-row preprocessing pipeline.
          1. Clean numeric fields
          2. Impute condition per-year
          3. Clean categoricals (title-case -> rare-map -> target encode)
          4. Engineer features
          5. Assemble in exact feature_cols order
        """
        car = dict(car)

        year      = self._clean_year(car.get('year'))
        odometer  = self._clean_odometer(car.get('odometer'))
        condition = self._impute_condition(car.get('condition'), year)

        if car.get('transmission') is None or \
            str(car.get('transmission')).strip().lower() in ('nan', 'none', ''):
            car['transmission'] = 'automatic'

        encoded = {}
        for col in CAT_COLS:
            cleaned_val          = self._clean_cat(col, car.get(col))
            encoded[col + '_te'] = self._target_encode(col, cleaned_val)

        #Feature engineering 
        car_age             = max(CURRENT_YEAR - year, 1)
        usage_intensity     = odometer / car_age
        condition_age_ratio = condition / car_age
        log_odometer        = np.log1p(odometer)

        # Assemble in exact feature_cols order
        row = {
            'year':                year,
            'car_age':             car_age,
            'odometer':            odometer,
            'log_odometer':        log_odometer,
            'condition':           condition,
            'usage_intensity':     usage_intensity,
            'condition_age_ratio': condition_age_ratio,
            **encoded
        }

        return np.array([[row[f] for f in self.feature_cols]])
    
    def place_bid(self, current_highest_bid: float) -> float:
        """
        Guards (checked in order):
            1. No prediction available
            2. Bankroll critically low
            3. Current bid already below minimum profit margin
            4. Current bid already at or above our ceiling
            5. Next bid would exceed ceiling -> bid at ceiling instead
            6. Next bid exceeds available bankroll
        """
        # Guard 1: no prediction
        if self.predicted_value is None or self.max_bid is None:
            return 0.0

        # Guard 2: bankroll critically low
        if self.bankroll / INITIAL_BANKROLL < BANKROLL_FLOOR_PCT:
            return 0.0

        # Guard 3: current bid already eats into minimum profit margin
        if current_highest_bid > 0:
            current_margin = (self.predicted_value - current_highest_bid) / self.predicted_value
            if current_margin < MIN_PROFIT_PCT:
                return 0.0

        # Guard 4: current bid already at or above our ceiling
        if current_highest_bid >= self.max_bid:
            return 0.0

        # Compute next bid
        increment = max(MIN_INCREMENT, current_highest_bid * INCREMENT_PCT)
        next_bid  = current_highest_bid + increment

        # If next bid overshoots ceiling, bid exactly at ceiling
        if next_bid > self.max_bid:
            next_bid = self.max_bid

        # Guard 6: can't afford it
        if next_bid > self.bankroll:
            return 0.0

        return round(next_bid, 2)


    def update_result(self, won: bool, final_price: float,
                      resale_value: float = None):
        """
        Called by the arena after each lot closes.

        Args:
            won          : True if we won this car
            final_price  : Hammer price paid
            resale_value : Actual resale value if provided by simulator;
                           falls back to predicted_value if not given
        """
        if won:
            self.bankroll    -= final_price
            self.cars_won    += 1
            self.total_spent += final_price

            estimated_resale  = resale_value if resale_value is not None \
                                else self.predicted_value
            self.total_profit += estimated_resale - final_price

        # Reset per-car state regardless of outcome
        self.predicted_value = None
        self.max_bid         = None


    def summary(self) -> dict:
        """Returns current session performance stats."""
        win_rate = self.cars_won / self.cars_seen if self.cars_seen > 0 else 0.0
        roi      = self.total_profit / self.total_spent if self.total_spent > 0 else 0.0
        return {
            'bankroll':     round(self.bankroll, 2),
            'cars_seen':    self.cars_seen,
            'cars_won':     self.cars_won,
            'win_rate_pct': round(win_rate * 100, 1),
            'total_spent':  round(self.total_spent, 2),
            'total_profit': round(self.total_profit, 2),
            'roi_pct':      round(roi * 100, 1),
            'net_position': round(self.bankroll - INITIAL_BANKROLL, 2),
        }