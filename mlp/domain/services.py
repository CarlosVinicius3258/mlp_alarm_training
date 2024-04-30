
class RuleBasedSystem:
    @staticmethod
    def determine_output(input_values):
        F, G, H, I, J = input_values
        if (not F) and H:
            return 1
        elif (not G) and H:
            return 1
        elif F and G and H:
            return 1
        elif (F and not G and not H) or (G and not F and not H) or (not F and not G and (I and not J)):
            return 1
        else:
            return 0