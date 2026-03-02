from typing import Callable, Optional

from .entity_stat import EntityStat


class EntityHunger(EntityStat):
    """
    Modela el estado de hambre de cualquier entidad del juego.

    Por convención, el valor máximo representa saciedad completa y el
    mínimo (0) representa hambre extrema. El ``regen_per_second`` suele
    ser negativo para simular el aumento gradual del hambre con el tiempo.

    Extiende EntityStat con semántica propia:
    - ``feed`` como alias expresivo de increase (alimentarse).
    - ``starve`` como alias de decrease (perder saciedad).
    - ``is_starving`` para detectar hambre extrema.
    - ``on_starve`` como alias de on_empty.
    """

    def __init__(
        self,
        maximum: float,
        current: Optional[float] = None,
        regen_per_second: float = 0.0,
        on_starve: Optional[Callable[[], None]] = None,
        on_change: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        """
        Args:
            maximum:          Saciedad máxima. Debe ser > 0.
            current:          Saciedad inicial. Por defecto igual a maximum.
            regen_per_second: Cambio automático por segundo (negativo = pierde saciedad).
            on_starve:        Llamado cuando la saciedad llega a 0.
            on_change:        Llamado con (valor_anterior, valor_nuevo) en cada cambio.
        """
        super().__init__(
            maximum=maximum,
            current=current,
            regen_per_second=regen_per_second,
            on_empty=on_starve,
            on_change=on_change,
        )

    # ------------------------------------------------------------------
    # Alias semánticos
    # ------------------------------------------------------------------

    @property
    def on_starve(self) -> Optional[Callable[[], None]]:
        return self.on_empty

    @on_starve.setter
    def on_starve(self, value: Optional[Callable[[], None]]) -> None:
        self.on_empty = value

    @property
    def is_starving(self) -> bool:
        """True cuando la saciedad es cero (hambre extrema)."""
        return self.is_empty

    def feed(self, amount: float) -> float:
        """
        Incrementa la saciedad en *amount* (alimentarse).

        Returns:
            Cantidad real ganada.
        """
        return self.increase(amount)

    def starve(self, amount: float) -> float:
        """
        Reduce la saciedad en *amount* (perder saciedad).

        Returns:
            Reducción real aplicada.
        """
        return self.decrease(amount)

