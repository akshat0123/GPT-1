
class RollingCounter:


    def __init__(self, limit: int):
        """ Counter keeping track of total and rolling average

        Args:
            limit: total size of "rolling" count
        """
        self.non_latest_total = 0
        self.non_latest_count = 0
        self.latest_values = []
        self.latest_total = 0
        self.limit = limit


    def add(self, x: int) -> None:
        """ Add number to rolling counts

        Args:
           x: number to add 
        """
        self.latest_values.append(x)
        self.latest_total += x

        if len(self.latest_values) > self.limit:
            remove = self.latest_values.pop(0)
            self.latest_total -= remove
            self.non_latest_total += remove
            self.non_latest_count += 1


    def total_average(self) -> float:
        count = self.non_latest_count + len(self.latest_values)
        total = self.non_latest_total + self.latest_total
        avg = total / count if count > 0 else 0
        return avg


    def rolling_average(self):
        total = self.latest_total
        count = len(self.latest_values)
        avg = total / count if count > 0 else 0
        return avg


