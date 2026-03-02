from enum import Enum, auto
from typing import Optional

from core.base.game_object import GameObject
from core.entity_stats.entity_health import EntityHealth


class ItemType(Enum):
    GENERIC    = auto()
    TOOL       = auto()
    WEAPON     = auto()
    AMMUNITION = auto()
    ARMOR      = auto()
    ACCESSORY  = auto()
    CONSUMABLE = auto()
    FURNITURE  = auto()

class ItemState(Enum):
    ON_GROUND    = auto()
    IN_INVENTORY = auto()
    EQUIPPED     = auto()


class BaseItem(GameObject):
    """Clase base para todos los ítems del juego."""

    ITEM_TYPE:          list[ItemType] = [ItemType.GENERIC]
    MAX_HEALTH:         float          = 100.0
    STACKABLE:          bool           = False   # True → varias unidades en un slot
    MAX_STACK:          int            = 1       # ignorado si STACKABLE es False
    INVENTORY_STORABLE: bool           = True    # False → no puede guardarse en un inventario (hoguera, estatua, etc.)

    def __init__(
        self,
        name: Optional[str] = None,
        description: str = "",
        value: int = 0,
        weight: float = 0.0,
    ) -> None:
        super().__init__(name=name)

        self.item_type:          list[ItemType]    = self.ITEM_TYPE
        self.description:        str               = description
        self.value:              int               = value
        self.weight:             float             = weight
        self.state:              ItemState         = ItemState.ON_GROUND
        self.owner:              Optional[object]  = None
        self.stack_count:        int               = 1
        self.inventory_storable: bool              = self.INVENTORY_STORABLE

        self.health: EntityHealth = EntityHealth(
            maximum=self.MAX_HEALTH,
            on_death=self.on_broken,
        )

    # ------------------------------------------------------------------
    # Apilado
    # ------------------------------------------------------------------

    @property
    def is_stackable(self) -> bool:
        """True si este ítem puede apilarse en un slot de inventario."""
        return self.STACKABLE

    @property
    def stack_space(self) -> int:
        """Unidades adicionales que caben en este slot (0 si está lleno o no es apilable)."""
        if not self.STACKABLE:
            return 0
        return self.MAX_STACK - self.stack_count

    def can_stack_with(self, other: "BaseItem") -> bool:
        """
        True si *other* puede apilarse en este slot.

        Condiciones: mismo tipo concreto, ambos apilables y slot con espacio.
        """
        return (
            self.STACKABLE
            and type(self) is type(other)
            and self.stack_space > 0
        )

    # ------------------------------------------------------------------
    # Salud
    # ------------------------------------------------------------------

    @property
    def is_broken(self) -> bool:
        """True si la salud del ítem ha llegado a 0."""
        return not self.health.is_alive

    def damage(self, amount: float) -> float:
        """
        Aplica daño a la durabilidad del ítem.

        Returns:
            Daño real aplicado.
        """
        return self.health.damage(amount)

    def repair(self, amount: float) -> float:
        """
        Repara el ítem en *amount* puntos.

        Returns:
            Cantidad real reparada.
        """
        return self.health.heal(amount)

    def on_broken(self) -> None:
        """
        Llamado cuando la salud del ítem llega a 0.
        Override en subclases para definir el comportamiento
        (destruir, perder efecto, cambiar estado, etc.).
        Por defecto destruye el ítem.
        """
        self.destroy()


    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def pick_up(self, actor) -> bool:
        if self.state != ItemState.ON_GROUND:
            return False
        self.state = ItemState.IN_INVENTORY
        self.owner = actor
        self.on_pick_up(actor)
        return True

    def drop(self, x: float, y: float) -> bool:
        if self.state == ItemState.ON_GROUND:
            return False
        prev = self.owner
        self.state = ItemState.ON_GROUND
        self.owner = None
        self.set_position(x, y)
        self.on_drop(prev)
        return True

    def use(self, user) -> bool:
        if not self.active:
            return False
        return self.on_use(user)

    def equip(self, actor) -> bool:
        if self.state != ItemState.IN_INVENTORY:
            return False
        self.state = ItemState.EQUIPPED
        self.on_equip(actor)
        return True

    def unequip(self) -> bool:
        if self.state != ItemState.EQUIPPED:
            return False
        self.state = ItemState.IN_INVENTORY
        self.on_unequip(self.owner)
        return True

    # ------------------------------------------------------------------
    # Hooks — override en subclases
    # ------------------------------------------------------------------

    def on_pick_up(self, actor) -> None:     pass
    def on_drop(self, prev_owner) -> None:   pass
    def on_equip(self, actor) -> None:       pass
    def on_unequip(self, actor) -> None:     pass

    def on_use(self, user) -> bool:
        """Override en subclases con la lógica de uso."""
        return False

    # ------------------------------------------------------------------
    # GameObject interface
    # ------------------------------------------------------------------

    def update(self, delta_time: float) -> None:
        self.health.update(delta_time)
    def render(self, renderer) -> None:          pass

    def __repr__(self) -> str:
        types = "|".join(t.name for t in self.item_type)
        stack = f" x{self.stack_count}" if self.STACKABLE else ""
        return (
            f"<{self.__class__.__name__} {self.name!r} "
            f"[{types}]{stack} {self.state.name}>"
        )
