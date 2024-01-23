

from dataclasses import dataclass, field

@dataclass
class Parameter:
    sizex: int
    sizey: int
    
    number_of_beads_per_strand: int
    number_of_strands: int
    
    contour_length_of_strand : float
    
    fix_boundary: bool = field(default=False) 

    @property 
    def Lx(self) -> float:
        return self.sizex / 2.0

    @property 
    def Ly(self) -> float:
        return self.sizex / 2.0