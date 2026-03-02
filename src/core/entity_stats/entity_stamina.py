from typing import Callable, Optional

from .entity_stat import EntityStat


class EntityStamina(EntityStat):
    """
    Modela la resistencia/stamina de cualquier entidad del juego.

    El valor máximo representa resistencia completa y el mínimo (0) agotamiento
    total. El ``regen_per_second`` suele ser positivo para recuperación pasiva,
    y se consume manualmente mediante ``exhaust``.

    Extiende EntityStat con semántica propia:
    - ``exhaust`` como alias expresivo de decrease (gastar stamina).
    - ``recover`` como alias de increase (recuperar stamina).
    - ``is_exhausted`` para detectar agotamiento total.
    - ``on_exhausted`` como alias de on_empty.
    """

    def __init__(
        self,
        maximum: float,
        current: Optional[float] = None,
        regen_per_second: float = 0.0,
        on_exhausted: Optional[Callable[[], None]] = None,
        on_change: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        """
        Args:
            maximum:          Stamina máxima. Debe ser > 0.
            current:          Stamina inicial. Por defecto igual a maximum.
            regen_per_second: Cambio automático por segundo (positivo = recuperación pasiva).
            on_exhausted:     Llamado cuando la stamina llega a 0.
            on_change:        Llamado con (valor_anterior, valor_nuevo) en cada cambio.
        """
        super().__init__(
            maximum=maximum,
            current=current,
            regen_per_second=regen_per_second,
            on_empty=on_exhausted,
            on_change=on_change,
        )

    # ------------------------------------------------------------------
    # Alias semánticos
    # ------------------------------------------------------------------

    @property
    def on_exhausted(self) -> Optional[Callable[[], None]]:
        return self.on_empty

    @on_exhausted.setter
    def on_exhausted(self, value: Optional[Callable[[], None]]) -> None:
        self.on_empty = value

    @property
    def is_exhausted(self) -> bool:
        """True cuando la stamina es cero."""
        return self.is_empty

    def exhaust(self, amount: float) -> float:
        """
        Reduce la stamina en *amount* (gastar energía).

        Returns:
            Reducción real aplicada.
        """
        return self.decrease(amount)

    def recover(self, amount: float) -> float:
        """
        Incrementa la stamina en *amount* (recuperar energía).

        Returns:
            Cantidad real recuperada.
        """
        return self.increase(amount)

