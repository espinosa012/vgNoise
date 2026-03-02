from typing import Callable, Optional


class EntityStat:
    """
    Clase base para estadísticas de entidad con valor numérico acotado
    (salud, hambre, resistencia, etc.).

    Soporta regeneración o decremento automático por frame mediante
    ``regen_per_second``: un valor positivo regenera, uno negativo drena.

    Attributes:
        current:          Valor actual.
        maximum:          Valor máximo.
        regen_per_second: Cambio automático por segundo (puede ser negativo).
        on_empty:         Callback opcional invocado cuando current llega a 0.
        on_change:        Callback opcional invocado con (prev, current) en
                          cada cambio de valor.
    """

    def __init__(
        self,
        maximum: float,
        current: Optional[float] = None,
        regen_per_second: float = 0.0,
        on_empty: Optional[Callable[[], None]] = None,
        on_change: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        """
        Args:
            maximum:          Valor máximo. Debe ser > 0.
            current:          Valor inicial. Por defecto igual a maximum.
            regen_per_second: Valor ganado/perdido por segundo automáticamente.
            on_empty:         Llamado cuando el valor llega a 0.
            on_change:        Llamado con (valor_anterior, valor_nuevo) en cada cambio.
        """
        if maximum <= 0:
            raise ValueError(f"maximum debe ser > 0, recibido: {maximum}")

        self.maximum:          float = maximum
        self._current:         float = current if current is not None else maximum
        self.regen_per_second: float = regen_per_second
        self.on_empty:         Optional[Callable[[], None]]             = on_empty
        self.on_change:        Optional[Callable[[float, float], None]] = on_change

        self._current = max(0.0, min(self._current, self.maximum))

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def current(self) -> float:
        return self._current

    @current.setter
    def current(self, value: float) -> None:
        self._set(value)

    @property
    def is_empty(self) -> bool:
        """True cuando el valor es cero."""
        return self._current <= 0.0

    @property
    def is_full(self) -> bool:
        """True cuando el valor está al máximo."""
        return self._current >= self.maximum

    @property
    def percentage(self) -> float:
        """Valor actual como fracción de 0.0 a 1.0."""
        return self._current / self.maximum

    # ------------------------------------------------------------------
    # Modificadores
    # ------------------------------------------------------------------

    def decrease(self, amount: float) -> float:
        """
        Reduce el valor en *amount* (debe ser positivo).

        Returns:
            Reducción real aplicada (puede ser menor si llega a 0).
        """
        amount = max(0.0, amount)
        prev = self._current
        self._set(self._current - amount)
        return prev - self._current

    def increase(self, amount: float) -> float:
        """
        Incrementa el valor en *amount* (debe ser positivo), sin superar maximum.

        Returns:
            Cantidad real incrementada (puede ser menor si ya estaba casi llena).
        """
        amount = max(0.0, amount)
        prev = self._current
        self._set(self._current + amount)
        return self._current - prev

    def set_maximum(self, new_max: float, scale_current: bool = False) -> None:
        """
        Cambia el valor máximo.

        Args:
            new_max:       Nuevo valor máximo (debe ser > 0).
            scale_current: Si True, escala el valor actual proporcionalmente.
        """
        if new_max <= 0:
            raise ValueError(f"new_max debe ser > 0, recibido: {new_max}")
        if scale_current:
            ratio = self._current / self.maximum
            self.maximum = new_max
            self._set(self.maximum * ratio)
        else:
            self.maximum = new_max
            self._set(self._current)  # re-clamp

    def reset(self) -> None:
        """Restaura el valor al máximo."""
        self._set(self.maximum)

    # ------------------------------------------------------------------
    # Update (regen/drain automático)
    # ------------------------------------------------------------------

    def update(self, delta_time: float) -> None:
        """
        Aplica la regeneración/drenaje automático según ``regen_per_second``.

        Debe llamarse cada frame con el tiempo transcurrido en segundos.

        Args:
            delta_time: Tiempo transcurrido desde el último frame en segundos.
        """
        if self.regen_per_second != 0.0 and not self.is_empty:
            self._set(self._current + self.regen_per_second * delta_time)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set(self, value: float) -> None:
        """Asigna el nuevo valor, lo clampea y dispara callbacks."""
        prev = self._current
        self._current = max(0.0, min(value, self.maximum))
        if self._current != prev:
            if self.on_change is not None:
                self.on_change(prev, self._current)
            if self._current == 0.0 and self.on_empty is not None:
                self.on_empty()

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(current={self._current:.1f}, "
            f"maximum={self.maximum:.1f}, "
            f"regen={self.regen_per_second:+.2f}/s)"
        )

