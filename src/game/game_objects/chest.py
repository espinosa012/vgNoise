from typing import Optional

import pygame

from core.item.item import BaseItem, ItemType
from core.item.inventory import Inventory


class BaseChest(BaseItem):
    """
    Cofre base: ítem de tipo FURNITURE con un inventario propio.

    Actúa como contenedor físico en el mundo del juego. Cualquier
    entidad con inventario (armario, caja, alforjas…) puede heredar
    de esta clase.

    Attributes:
        inventory: Inventario que gestiona los ítems almacenados.
        sprite:    Surface de pygame que representa visualmente el cofre.
                   None hasta que se asigne o cargue un recurso gráfico.
        is_open:   Estado de apertura del cofre.
    """

    ITEM_TYPE:  list[ItemType] = [ItemType.FURNITURE]
    MAX_HEALTH: float          = 200.0
    STACKABLE:  bool           = False

    def __init__(
        self,
        name: Optional[str] = None,
        description: str = "",
        value: int = 0,
        weight: float = 10.0,
        inventory_capacity: Optional[int] = 20,
        inventory_max_weight: Optional[float] = None,
        allowed_types: Optional[list[ItemType]] = None,
        sprite: Optional[pygame.Surface] = None,
    ) -> None:
        """
        Args:
            name:                  Nombre del cofre.
            description:           Descripción visible en la UI.
            value:                 Valor de venta/compra.
            weight:                Peso del cofre vacío.
            inventory_capacity:    Slots máximos del inventario interno.
            inventory_max_weight:  Peso máximo que soporta el inventario.
            allowed_types:         Tipos de ítem aceptados. None → todos.
            sprite:                Surface de pygame para renderizarlo.
        """
        super().__init__(
            name=name,
            description=description,
            value=value,
            weight=weight,
        )

        self.inventory: Inventory = Inventory(
            capacity=inventory_capacity,
            max_weight=inventory_max_weight,
            allowed_types=allowed_types,
        )

        self.sprite:   Optional[pygame.Surface] = sprite
        self.is_open:  bool                     = False

    # ------------------------------------------------------------------
    # Apertura / cierre
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Abre el cofre."""
        self.is_open = True
        self.on_open()

    def close(self) -> None:
        """Cierra el cofre."""
        self.is_open = False
        self.on_close()

    def toggle(self) -> None:
        """Alterna entre abierto y cerrado."""
        self.open() if not self.is_open else self.close()

    # ------------------------------------------------------------------
    # Hooks — override en subclases
    # ------------------------------------------------------------------

    def on_open(self) -> None:
        """Llamado al abrir el cofre. Override para lógica extra (sonido, animación…)."""
        pass

    def on_close(self) -> None:
        """Llamado al cerrar el cofre."""
        pass

    def on_broken(self) -> None:
        """Al romperse el cofre, sus ítems caen al suelo."""
        for item in self.inventory.items:
            self.inventory.drop(item, self.x, self.y)
        super().on_broken()

    # ------------------------------------------------------------------
    # BaseItem interface
    # ------------------------------------------------------------------

    def on_use(self, user) -> bool:
        """Usar el cofre equivale a abrirlo/cerrarlo."""
        self.toggle()
        return True

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state = "abierto" if self.is_open else "cerrado"
        return (
            f"<{self.__class__.__name__} {self.name!r} "
            f"[{state}] {self.inventory}>"
        )

