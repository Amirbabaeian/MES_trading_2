"""
Unit Tests for Futures Contract Configuration and Roll Logic

Tests cover:
- FuturesContract specification (MES, ES, etc.)
- ContractMonths enumeration and abbreviation lookup
- TradingHours configuration
- Contract code generation and parsing
- ContractRegistry for managing multiple contracts
- Contract roll date calculations
- Active contract determination
- Roll schedule management
"""

import pytest
from datetime import datetime, date, time
from src.backtest.contracts import (
    FuturesContract, ContractMonths, TradingHours, ContractRegistry
)
from src.backtest.config.mes_specs import (
    MES_CONTRACT, ES_CONTRACT, NQ_CONTRACT, YM_CONTRACT
)
from src.backtest.utils.rolls import (
    calculate_roll_dates, get_active_contract, RollSchedule,
    get_contract_expiration_date, calculate_first_notice_date,
    get_next_contract_month
)


# ============================================================================
# FuturesContract Tests
# ============================================================================

class TestFuturesContract:
    """Test FuturesContract configuration class."""
    
    def test_mes_contract_specifications(self):
        """Verify MES contract meets CME specifications."""
        # MES specifications per CME
        assert MES_CONTRACT.symbol == 'MES'
        assert MES_CONTRACT.name == 'Micro E-mini S&P 500'
        assert MES_CONTRACT.exchange == 'CME (GLOBEX)'
        assert MES_CONTRACT.multiplier == 5.0  # $5 per index point
        assert MES_CONTRACT.tick_size == 0.25  # 0.25 index points
        assert MES_CONTRACT.tick_value == 1.25  # 0.25 × $5
        assert MES_CONTRACT.initial_margin == 500.0
        assert MES_CONTRACT.maintenance_margin == 400.0
        assert MES_CONTRACT.price_precision == 2  # Two decimal places
    
    def test_es_contract_specifications(self):
        """Verify ES contract specifications."""
        assert ES_CONTRACT.symbol == 'ES'
        assert ES_CONTRACT.multiplier == 50.0  # 10× MES
        assert ES_CONTRACT.tick_size == 0.25
        assert ES_CONTRACT.tick_value == 12.50  # 0.25 × $50
        assert ES_CONTRACT.initial_margin == 12500.0
    
    def test_contract_months(self):
        """Verify quarterly contract months."""
        assert ContractMonths.MARCH in MES_CONTRACT.contract_months
        assert ContractMonths.JUNE in MES_CONTRACT.contract_months
        assert ContractMonths.SEPTEMBER in MES_CONTRACT.contract_months
        assert ContractMonths.DECEMBER in MES_CONTRACT.contract_months
        assert len(MES_CONTRACT.contract_months) == 4
    
    def test_contract_code_generation(self):
        """Test contract code generation from month and year."""
        # March 2024 -> MESH4
        code = MES_CONTRACT.get_contract_code(ContractMonths.MARCH, 2024)
        assert code == 'MESH4'
        
        # June 2024 -> MESM4
        code = MES_CONTRACT.get_contract_code(ContractMonths.JUNE, 2024)
        assert code == 'MESM4'
        
        # December 2024 -> MESZ4
        code = MES_CONTRACT.get_contract_code(ContractMonths.DECEMBER, 2024)
        assert code == 'MESZ4'
    
    def test_contract_code_parsing(self):
        """Test parsing contract codes back to month/year."""
        # MESH4 -> March 2024
        month, year = MES_CONTRACT.parse_contract_code('MESH4')
        assert month == ContractMonths.MARCH
        assert year == 2024
        
        # MESM4 -> June 2024
        month, year = MES_CONTRACT.parse_contract_code('MESM4')
        assert month == ContractMonths.JUNE
        assert year == 2024
        
        # MESZ25 -> December 2025
        month, year = MES_CONTRACT.parse_contract_code('MESZ25')
        assert month == ContractMonths.DECEMBER
        assert year == 2025
    
    def test_invalid_contract_code_parsing(self):
        """Test parsing invalid contract codes."""
        assert MES_CONTRACT.parse_contract_code('INVALID') is None
        assert MES_CONTRACT.parse_contract_code('MESH') is None
        assert MES_CONTRACT.parse_contract_code('ESH4') is None  # Wrong symbol
    
    def test_tick_value_calculation(self):
        """Test tick value is correctly calculated."""
        # MES: $1.25 per tick
        assert MES_CONTRACT.tick_value == pytest.approx(1.25)
        # ES: $12.50 per tick
        assert ES_CONTRACT.tick_value == pytest.approx(12.50)
        # NQ: 0.25 × $20 = $5.00 per tick
        assert NQ_CONTRACT.tick_value == pytest.approx(5.0)
    
    def test_price_precision(self):
        """Test price precision is calculated from tick size."""
        # tick_size=0.25 -> 2 decimal places
        assert MES_CONTRACT.price_precision == 2
        assert ES_CONTRACT.price_precision == 2
        # tick_size=1.0 -> 1 decimal place
        assert YM_CONTRACT.price_precision == 1
    
    def test_month_code_lookup(self):
        """Test getting month code from enum."""
        assert ContractMonths.MARCH.abbreviation == 'H'
        assert ContractMonths.JUNE.abbreviation == 'M'
        assert ContractMonths.SEPTEMBER.abbreviation == 'U'
        assert ContractMonths.DECEMBER.abbreviation == 'Z'
    
    def test_month_abbreviation_lookup(self):
        """Test getting month enum from abbreviation."""
        assert ContractMonths.from_abbreviation('H') == ContractMonths.MARCH
        assert ContractMonths.from_abbreviation('M') == ContractMonths.JUNE
        assert ContractMonths.from_abbreviation('U') == ContractMonths.SEPTEMBER
        assert ContractMonths.from_abbreviation('Z') == ContractMonths.DECEMBER
        assert ContractMonths.from_abbreviation('X') is None  # Invalid
    
    def test_contract_validation(self):
        """Test contract validation during initialization."""
        # Valid contract should not raise
        contract = FuturesContract(
            symbol='TEST',
            name='Test Contract',
            exchange='TEST',
            multiplier=1.0,
            tick_size=0.01,
            initial_margin=100.0,
        )
        assert contract is not None
        
        # Invalid: negative multiplier
        with pytest.raises(ValueError):
            FuturesContract(
                symbol='TEST',
                name='Test',
                exchange='TEST',
                multiplier=-1.0,
                tick_size=0.01,
                initial_margin=100.0,
            )
        
        # Invalid: empty symbol
        with pytest.raises(ValueError):
            FuturesContract(
                symbol='',
                name='Test',
                exchange='TEST',
                multiplier=1.0,
                tick_size=0.01,
                initial_margin=100.0,
            )
    
    def test_contract_serialization(self):
        """Test contract to_dict and from_dict."""
        spec_dict = MES_CONTRACT.to_dict()
        
        # Verify all key fields are in dict
        assert spec_dict['symbol'] == 'MES'
        assert spec_dict['multiplier'] == 5.0
        assert spec_dict['tick_size'] == 0.25
        assert spec_dict['tick_value'] == 1.25
        assert spec_dict['initial_margin'] == 500.0
        assert spec_dict['maintenance_margin'] == 400.0
        
        # Test round-trip
        reconstructed = FuturesContract.from_dict(spec_dict)
        assert reconstructed.symbol == MES_CONTRACT.symbol
        assert reconstructed.multiplier == MES_CONTRACT.multiplier
        assert reconstructed.tick_size == MES_CONTRACT.tick_size


# ============================================================================
# TradingHours Tests
# ============================================================================

class TestTradingHours:
    """Test TradingHours configuration."""
    
    def test_trading_hours_creation(self):
        """Test creating trading hours."""
        hours = TradingHours(
            open_time=time(9, 30),
            close_time=time(16, 0),
            name="Regular Hours",
        )
        assert hours.open_time == time(9, 30)
        assert hours.close_time == time(16, 0)
        assert hours.name == "Regular Hours"
    
    def test_globex_hours(self):
        """Test Globex nearly 24-hour trading."""
        globex_hours = MES_CONTRACT.trading_hours
        assert globex_hours.open_time == time(17, 0)  # Sunday 5 PM CT
        assert globex_hours.close_time == time(16, 0)  # Friday 4 PM CT
        assert globex_hours.timezone == "America/Chicago"
    
    def test_trading_hours_serialization(self):
        """Test trading hours to_dict and from_dict."""
        hours = MES_CONTRACT.trading_hours
        hours_dict = hours.to_dict()
        
        reconstructed = TradingHours.from_dict(hours_dict)
        assert reconstructed.open_time == hours.open_time
        assert reconstructed.close_time == hours.close_time
        assert reconstructed.timezone == hours.timezone


# ============================================================================
# ContractRegistry Tests
# ============================================================================

class TestContractRegistry:
    """Test ContractRegistry for managing multiple contracts."""
    
    def test_registry_creation(self):
        """Test creating empty registry."""
        registry = ContractRegistry()
        assert len(registry) == 0
        assert registry.list_symbols() == []
    
    def test_register_contract(self):
        """Test registering contracts."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        registry.register(ES_CONTRACT)
        
        assert len(registry) == 2
        assert 'MES' in registry
        assert 'ES' in registry
    
    def test_get_contract(self):
        """Test retrieving registered contracts."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        
        retrieved = registry.get('MES')
        assert retrieved is MES_CONTRACT
        assert retrieved.symbol == 'MES'
        
        # Non-existent contract
        assert registry.get('NONEXISTENT') is None
    
    def test_list_symbols(self):
        """Test listing all registered symbols."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        registry.register(ES_CONTRACT)
        registry.register(NQ_CONTRACT)
        
        symbols = registry.list_symbols()
        assert symbols == ['ES', 'MES', 'NQ']  # Sorted
    
    def test_list_contracts(self):
        """Test listing all registered contracts."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        registry.register(ES_CONTRACT)
        
        contracts = registry.list_contracts()
        assert len(contracts) == 2
        symbols = [c.symbol for c in contracts]
        assert 'MES' in symbols
        assert 'ES' in symbols
    
    def test_duplicate_registration(self):
        """Test that duplicate symbol raises error."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        
        with pytest.raises(ValueError):
            registry.register(MES_CONTRACT)
    
    def test_register_or_update(self):
        """Test register_or_update allows overwriting."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        
        # Update should succeed
        registry.register_or_update(MES_CONTRACT)
        assert len(registry) == 1
    
    def test_unregister(self):
        """Test unregistering contracts."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        registry.register(ES_CONTRACT)
        
        success = registry.unregister('MES')
        assert success is True
        assert 'MES' not in registry
        assert len(registry) == 1
        
        # Unregister non-existent
        success = registry.unregister('NONEXISTENT')
        assert success is False
    
    def test_clear(self):
        """Test clearing all contracts."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        registry.register(ES_CONTRACT)
        
        registry.clear()
        assert len(registry) == 0
    
    def test_contains_operator(self):
        """Test 'in' operator."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        
        assert 'MES' in registry
        assert 'ES' not in registry
    
    def test_iteration(self):
        """Test iterating over symbols."""
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        registry.register(ES_CONTRACT)
        
        symbols = list(registry)
        assert symbols == ['ES', 'MES']


# ============================================================================
# Contract Roll Tests
# ============================================================================

class TestContractRolls:
    """Test contract roll date calculations."""
    
    def test_contract_expiration_date(self):
        """Test calculating contract expiration dates."""
        # March 2024 expiration should be a Friday
        exp = get_contract_expiration_date(MES_CONTRACT, ContractMonths.MARCH, 2024)
        assert exp.month == 3
        assert exp.year == 2024
        assert exp.weekday() == 4  # Friday
    
    def test_first_notice_date(self):
        """Test first notice date calculation."""
        fnd = calculate_first_notice_date(MES_CONTRACT, ContractMonths.MARCH, 2024)
        exp = get_contract_expiration_date(MES_CONTRACT, ContractMonths.MARCH, 2024)
        
        # FND should be before expiration
        assert fnd < exp
        # Standard is 5 days before
        delta = (exp - fnd).days
        assert delta == 5
    
    def test_next_contract_month(self):
        """Test getting next contract month in sequence."""
        months = [ContractMonths.MARCH, ContractMonths.JUNE,
                  ContractMonths.SEPTEMBER, ContractMonths.DECEMBER]
        
        # March -> June (same year)
        next_month, offset = get_next_contract_month(ContractMonths.MARCH, months)
        assert next_month == ContractMonths.JUNE
        assert offset == 0
        
        # December -> March (next year)
        next_month, offset = get_next_contract_month(ContractMonths.DECEMBER, months)
        assert next_month == ContractMonths.MARCH
        assert offset == 1
    
    def test_calculate_roll_dates_single_year(self):
        """Test calculating all roll dates for a year."""
        rolls = calculate_roll_dates(MES_CONTRACT, 2024)
        
        # Should have one roll per contract month (quarterly)
        assert len(rolls) == 4
        
        # Verify order
        assert rolls[0].from_month == ContractMonths.MARCH
        assert rolls[1].from_month == ContractMonths.JUNE
        assert rolls[2].from_month == ContractMonths.SEPTEMBER
        assert rolls[3].from_month == ContractMonths.DECEMBER
        
        # Verify roll dates are chronological
        for i in range(len(rolls) - 1):
            assert rolls[i].roll_date < rolls[i + 1].roll_date
    
    def test_roll_contract_codes(self):
        """Test contract codes in rolls."""
        rolls = calculate_roll_dates(MES_CONTRACT, 2024)
        
        # First roll: from MESH4 to MESM4
        assert rolls[0].from_contract == 'MESH4'
        assert rolls[0].to_contract == 'MESM4'
        
        # Last roll: from MESZ4 to MESH5
        assert rolls[3].from_contract == 'MESZ4'
        assert rolls[3].to_contract == 'MESH5'
    
    def test_roll_year_boundary(self):
        """Test rolls crossing year boundary."""
        rolls = calculate_roll_dates(MES_CONTRACT, 2024)
        
        # December 2024 to March 2025 roll
        last_roll = rolls[3]
        assert last_roll.from_year == 2024
        assert last_roll.to_year == 2025
        assert last_roll.is_year_boundary_roll is True
    
    def test_get_active_contract(self):
        """Test determining active contract at a timestamp."""
        # January 2024 -> MARCH contract active
        dt = datetime(2024, 1, 15, 10, 0)
        month, year = get_active_contract(MES_CONTRACT, dt)
        assert month == ContractMonths.MARCH
        assert year == 2024
        
        # July 2024 -> SEPTEMBER contract active
        dt = datetime(2024, 7, 15, 10, 0)
        month, year = get_active_contract(MES_CONTRACT, dt)
        assert month == ContractMonths.SEPTEMBER
        assert year == 2024
        
        # December 2024 -> next MARCH (2025)
        dt = datetime(2024, 12, 15, 10, 0)
        month, year = get_active_contract(MES_CONTRACT, dt)
        assert month == ContractMonths.MARCH
        assert year == 2025


# ============================================================================
# RollSchedule Tests
# ============================================================================

class TestRollSchedule:
    """Test RollSchedule for managing rolls over extended periods."""
    
    def test_roll_schedule_creation(self):
        """Test creating roll schedule."""
        schedule = RollSchedule(
            contract=MES_CONTRACT,
            start_year=2024,
            end_year=2024,
        )
        
        # Should have 4 rolls for one year (quarterly)
        assert len(schedule.rolls) == 4
    
    def test_roll_schedule_multiple_years(self):
        """Test roll schedule spanning multiple years."""
        schedule = RollSchedule(
            contract=MES_CONTRACT,
            start_year=2024,
            end_year=2025,
        )
        
        # Should have 8 rolls (4 per year)
        assert len(schedule.rolls) == 8
        # All rolls sorted by date
        for i in range(len(schedule.rolls) - 1):
            assert schedule.rolls[i].roll_date < schedule.rolls[i + 1].roll_date
    
    def test_get_active_contract_at(self):
        """Test finding active contract from schedule."""
        schedule = RollSchedule(
            contract=MES_CONTRACT,
            start_year=2024,
            end_year=2024,
        )
        
        # Before first roll
        dt = datetime(2024, 1, 1, 10, 0)
        active = schedule.get_active_contract_at(dt)
        assert active is None
        
        # After first roll
        dt = datetime(2024, 2, 1, 10, 0)
        active = schedule.get_active_contract_at(dt)
        assert active is not None
        assert active.from_contract == 'MESH4'
    
    def test_get_upcoming_rolls(self):
        """Test getting upcoming rolls."""
        schedule = RollSchedule(
            contract=MES_CONTRACT,
            start_year=2024,
            end_year=2025,
        )
        
        dt = datetime(2024, 1, 1, 10, 0)
        upcoming = schedule.get_upcoming_rolls(dt, num_rolls=2)
        
        assert len(upcoming) == 2
        assert upcoming[0].roll_date > dt
        assert upcoming[1].roll_date > dt
        assert upcoming[0].roll_date < upcoming[1].roll_date
    
    def test_days_until_next_roll(self):
        """Test calculating days until next roll."""
        schedule = RollSchedule(
            contract=MES_CONTRACT,
            start_year=2024,
            end_year=2024,
        )
        
        # Get first roll date
        first_roll = schedule.rolls[0]
        
        # Check days until roll from before it
        dt = first_roll.roll_date - timedelta(days=10)
        days = schedule.days_until_next_roll(dt)
        assert 9 < days < 11  # Should be approximately 10 days
        
        # Check days until roll from after all rolls
        dt = schedule.rolls[-1].roll_date + timedelta(days=100)
        days = schedule.days_until_next_roll(dt)
        assert days == float('inf')  # No more upcoming rolls


# ============================================================================
# Integration Tests
# ============================================================================

class TestBacktestConfigIntegration:
    """Test integration with BacktestConfig."""
    
    def test_get_contract_registry(self):
        """Test getting contract registry from config."""
        from src.backtest.config import BacktestConfig
        
        config = BacktestConfig()
        registry = config.get_contract_registry()
        
        # Should have standard contracts
        assert 'MES' in registry
        assert 'ES' in registry
        assert 'NQ' in registry
        assert 'YM' in registry
        
        # Verify specs match
        mes = registry.get('MES')
        assert mes.multiplier == 5.0
        assert mes.tick_value == 1.25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
