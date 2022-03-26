
class RollingCounter:


    def __init__(self, limit: int):
        self.non_latest_total = 0
        self.non_latest_count = 0
        self.latest_values = []
        self.latest_total = 0
        self.limit = limit


    def add(self, x: int):
        self.latest_values.append(x)
        self.latest_total += x

        if len(self.latest_values) > self.limit:
            remove = self.latest_values.pop(0)
            self.latest_total -= remove
            self.non_latest_total += remove
            self.non_latest_count += 1


    def total_average(self):
        count = self.non_latest_count + len(self.latest_values)
        total = self.non_latest_total + self.latest_total
        return total / count


    def rolling_average(self):
        return self.latest_total / len(self.latest_values)


