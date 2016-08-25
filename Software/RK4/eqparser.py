import re

class EquationSystem():
    def __init__(self,equations):

        self.raw_equations = equations
        self.left_hand_sides = []
        self.right_hand_sides = []

        self.independent_variable = None
        self.derivarives = []


        self.left_form1 = re.compile(r"^\s*(.)\'\s*$")
        self.left_form2 = re.compile(r"^\s*(.)\'\((.)\)\s*$")

    def _separe_sides(self):

        for equation in self.raw_equations:
            left, right = equation.split('=')
            self.left_hand_sides.append(left)
            self.right_hand_sides.append(right)

        assert( len(self.left_hand_sides) == len(self.right_hand_sides) )

    def _process_left_hand_sides(self):

        for left_side in self.left_hand_sides:

            form1 = self.left_form1.match(left_side)
            form2 = self.left_form2.match(left_side)

            if form1:
                self.derivarives.append( form1.group(1) )
                independent_variable = 'x'
            elif form2:
                self.derivarives.append(form2.group(1))
                independent_variable = form2.group(2)
            else:
                raise RuntimeError("""
                Invalid left hand side: {}.

                The left hand side must be one of the two following forms:

                    - __var__'
                    - __var__'( __independent_var__ )
                """.format(left_side))

            if self.independent_variable is None:
                self.independent_variable = independent_variable
            else:
                assert( self.independent_variable == independent_variable )

    def _transform_right_hand_sides(self):

        transform_map = {variable:'__var__[{}]'.format(index) for index, variable in enumerate(self.derivarives) }
        transform_map.update({self.independent_variable:'x'})
        for index,right_side in enumerate(self.right_hand_sides):
            new_right_side = right_side
            for variable,replacement in transform_map.items():
                new_right_side = new_right_side.replace( variable, replacement)
            #Delete any (x) floating around
            new_right_side = new_right_side.replace("](x)","]")
            yield new_right_side

    def parse(self):

        self._separe_sides()
        self._process_left_hand_sides()
        return list( self._transform_right_hand_sides() )
