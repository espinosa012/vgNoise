from typing import Optional

from core.item.item import BaseItem, ItemType


class Inventory:
    """
    Componente de inventario para cualquier entidad del juego
    (cofre, mochila, alforjas, armario, etc.).

    No es un objeto físico del mundo: es una característica que se
    compone dentro de otra entidad.

    Attributes:
        capacity:      Número máximo de slots ocupados.
                       None → sin límite de slots.
        max_weight:    Peso máximo acumulado.
                       None → sin límite de peso.
        allowed_types: Tipos de ítem aceptados.
                       None → acepta cualquier tipo.
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        max_weight: Optional[float] = None,
        allowed_types: Optional[list[ItemType]] = None,
    ) -> None:
        self.capacity:      Optional[int]           = capacity
        self.max_weight:    Optional[float]         = max_weight
        self.allowed_types: Optional[set[ItemType]] = (
            set(allowed_types) if allowed_types else None
        )
        self._items: list[BaseItem] = []

    # ------------------------------------------------------------------
    # Propiedades de estado
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[BaseItem]:
        """Lista de slots (copia de sólo lectura)."""
        return list(self._items)

    @property
    def count(self) -> int:
        """Número de slots ocupados."""
        return len(self._items)

    @property
    def current_weight(self) -> float:
        """Suma del weight de todos los ítems (weight × stack_count)."""
        return sum(item.weight * item.stack_count for item in self._items)

    @property
    def is_full(self) -> bool:
        """
        True si no hay espacio para ningún ítem adicional.

        No está lleno si hay algún slot apilable con espacio disponible,
        aunque el número de slots esté al máximo.
        """
        if self.capacity is None:
            return False
        if self.count < self.capacity:
            return False
        # Slots al máximo: sigue habiendo hueco si algún stack no está lleno
        return not any(item.stack_space > 0 for item in self._items)

    @property
    def is_empty(self) -> bool:
        return len(self._items) == 0

    @property
    def remaining_slots(self) -> Optional[int]:
        """Slots libres. None si no hay límite de capacidad."""
        if self.capacity is None:
            return None
        return self.capacity - self.count

    @property
    def remaining_weight(self) -> Optional[float]:
        """Peso disponible. None si no hay límite de peso."""
        if self.max_weight is None:
            return None
        return max(0.0, self.max_weight - self.current_weight)

    # ------------------------------------------------------------------
    # Validaciones internas
    # ------------------------------------------------------------------

    def accepts_type(self, item: BaseItem) -> bool:
        """True si el tipo del ítem está permitido en este inventario."""
        if not self.allowed_types:
            return True
        return bool(self.allowed_types.intersection(item.item_type))

    def _find_stack_slot(self, item: BaseItem) -> Optional[BaseItem]:
        """
        Devuelve el primer slot existente en el que *item* puede apilarse,
        o None si no existe ninguno.
        """
        for slot in self._items:
            if slot.can_stack_with(item):
                return slot
        return None

    def can_add(self, item: BaseItem) -> bool:
        """
        Comprueba si el ítem puede añadirse sin modificar el inventario.

        Primero intenta apilarlo en un slot existente compatible; si no es
        posible, verifica que haya un slot libre.
        """
        if not self.accepts_type(item):
            return False
        if (self.max_weight is not None
                and self.current_weight + item.weight > self.max_weight):
            return False
        # Cabe en un stack existente → no necesita slot nuevo
        if item.is_stackable and self._find_stack_slot(item) is not None:
            return True
        # Necesita slot nuevo
        if item in self._items:
            return False
        if self.capacity is not None and self.count >= self.capacity:
            return False
        return True

    # ------------------------------------------------------------------
    # Operaciones
    # ------------------------------------------------------------------

    def add(self, item: BaseItem) -> bool:
        """
        Añade el ítem al inventario.

        Si el ítem es apilable y hay un slot compatible con espacio,
        incrementa su stack_count en lugar de ocupar un slot nuevo.
        En caso contrario, crea un slot nuevo y llama a item.pick_up(self).

        Returns:
            True si se añadió o apiló correctamente.
        """
        if not self.can_add(item):
            return False

        if item.is_stackable:
            slot = self._find_stack_slot(item)
            if slot is not None:
                slot.stack_count += 1
                return True

        item.pick_up(self)
        self._items.append(item)
        return True

    def remove(self, item: BaseItem) -> bool:
        """
        Elimina el ítem del inventario sin posicionarlo en el mundo.
        Usa drop() si quieres colocarlo en una posición del mapa.

        Returns:
            True si el ítem estaba en el inventario y se eliminó.
        """
        if item not in self._items:
            return False
        self._items.remove(item)
        return True

    def drop(self, item: BaseItem, x: float, y: float) -> bool:
        """
        Extrae el ítem del inventario y lo coloca en (x, y) del mundo.

        Returns:
            True si la operación tuvo éxito.
        """
        if item not in self._items:
            return False
        self._items.remove(item)
        item.drop(x, y)
        return True

    def get_by_type(self, item_type: ItemType) -> list[BaseItem]:
        """Devuelve todos los slots cuyo ítem contiene el tipo dado."""
        return [i for i in self._items if item_type in i.item_type]

    def get_by_name(self, name: str) -> Optional[BaseItem]:
        """Devuelve el primer slot cuyo nombre coincida exactamente."""
        for item in self._items:
            if item.name == name:
                return item
        return None

    def clear(self) -> None:
        """Vacía el inventario sin llamar a ningún hook sobre los ítems."""
        self._items.clear()

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: BaseItem) -> bool:
        return item in self._items

    def __iter__(self):
        return iter(self._items)

    def __repr__(self) -> str:
        cap = f"{self.count}/{self.capacity}" if self.capacity else str(self.count)
        w = (f"{self.current_weight:.1f}/{self.max_weight}"
             if self.max_weight else f"{self.current_weight:.1f}")
        return f"Inventory(slots={cap}, weight={w})"

