from typing import Callable, Optional

from .entity_stat import EntityStat


class EntityHealth(EntityStat):
    """
    Modela el estado de salud de cualquier entidad del juego
    (personaje, ítem, estructura, etc.).

    Extiende EntityStat con semántica propia de salud:
    - ``damage`` / ``heal`` como alias expresivos de decrease/increase.
    - ``is_alive`` para comprobar si la entidad sigue en pie.
    - ``on_death`` como alias de on_empty para mantener legibilidad.
    """

    def __init__(
        self,
        maximum: float,
        current: Optional[float] = None,
        regen_per_second: float = 0.0,
        on_death: Optional[Callable[[], None]] = None,
        on_change: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        """
        Args:
            maximum:          Salud máxima. Debe ser > 0.
            current:          Salud inicial. Por defecto igual a maximum.
            regen_per_second: Salud ganada/perdida por segundo automáticamente.
            on_death:         Llamado cuando la salud llega a 0.
            on_change:        Llamado con (valor_anterior, valor_nuevo) en cada cambio.
        """
        super().__init__(
            maximum=maximum,
            current=current,
            regen_per_second=regen_per_second,
            on_empty=on_death,
            on_change=on_change,
        )

    # ------------------------------------------------------------------
    # Alias semánticos
    # ------------------------------------------------------------------

    @property
    def on_death(self) -> Optional[Callable[[], None]]:
        return self.on_empty

    @on_death.setter
    def on_death(self, value: Optional[Callable[[], None]]) -> None:
        self.on_empty = value

    @property
    def is_alive(self) -> bool:
        """True mientras la salud sea mayor que cero."""
        return not self.is_empty

    def damage(self, amount: float) -> float:
        """
        Reduce la salud en *amount* (debe ser positivo).

        Returns:
            Daño real aplicado (puede ser menor si la salud llega a 0).
        """
        return self.decrease(amount)

    def heal(self, amount: float) -> float:
        """
        Incrementa la salud en *amount* (debe ser positivo), sin superar maximum.

        Returns:
            Cantidad real curada (puede ser menor si ya estaba casi llena).
        """
        return self.increase(amount)

