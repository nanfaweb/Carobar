import json
import logging
import sys
import re
from datetime import datetime, timedelta

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLEANER] %(message)s")
log = logging.getLogger(__name__)

class CarDataCleaner:
    def __init__(self):
        self.seen_ids = set()

    def clean_text(self, text):
        """Collapses multiple internal spaces and trims ends."""
        if not text:
            return None
        # Replace all whitespace (\s+) with a single space
        text = re.sub(r'\s+', ' ', str(text))
        return text.strip().title()

    def clean_engine_cc(self, cc_text):
        """Extracts numbers from strings like '4600cc'."""
        if not cc_text:
            return None
        numbers = re.findall(r'\d+', str(cc_text))
        return int("".join(numbers)) if numbers else None

    def parse_updated_at(self, text):
        """Converts 'Updated X ago' into a real ISO timestamp."""
        if not text or not isinstance(text, str):
            return None
        
        now = datetime.now()
        text = text.lower()
        
        try:
            if "minute" in text:
                minutes = re.search(r'\d+', text)
                val = int(minutes.group()) if minutes else 1
                return (now - timedelta(minutes=val)).isoformat()
            elif "hour" in text:
                hours = re.search(r'\d+', text)
                val = int(hours.group()) if hours else 1
                return (now - timedelta(hours=val)).isoformat()
            elif "day" in text:
                days = re.search(r'\d+', text)
                val = int(days.group()) if days else 1
                return (now - timedelta(days=val)).isoformat()
        except Exception:
            pass
        return now.isoformat()

    def clean_record(self, car):
        try:
            # 1. Deduplication
            l_id = car.get("listing_id")
            if not l_id or l_id in self.seen_ids:
                return None
            self.seen_ids.add(l_id)

            # 2. Mandatory Price Check
            if not car.get("price_pkr") or car.get("price_pkr") == 0:
                return None

            # 3. Text & Engine Cleaning
            car["title"] = self.clean_text(car.get("title"))
            car["make"] = self.clean_text(car.get("make"))
            car["engine_cc"] = self.clean_engine_cc(car.get("engine_cc"))
            car["location"] = self.clean_text(car.get("location"))
            
            # 4. FIX: Handle the 'Updated at' timestamp error
            car["updated_at"] = self.parse_updated_at(car.get("updated_at"))

            # 5. Year Validation
            year = str(car.get("year", ""))
            car["year"] = int(year) if re.match(r"^(19|20)\d{2}$", year) else None

            # 6. Model Cleaning
            raw_model = car.get("model")
            if car["make"] and raw_model:
                clean_model = str(raw_model).replace(car["make"], "")
                car["model"] = self.clean_text(clean_model)
            else:
                car["model"] = self.clean_text(raw_model)

            return car
        except Exception as e:
            log.error(f"Error cleaning record: {e}")
            return None

    def process_list(self, raw_data):
        cleaned_data = []
        for record in raw_data:
            cleaned = self.clean_record(record)
            if cleaned:
                cleaned_data.append(cleaned)
        log.info(f"Cleaning complete. Output: {len(cleaned_data)}")
        return cleaned_data

if __name__ == "__main__":
    try:
        if not sys.stdin.isatty():
            raw_input = sys.stdin.read()
            data = json.loads(raw_input)
            cleaner = CarDataCleaner()
            print(json.dumps(cleaner.process_list(data)))
    except Exception as e:
        log.error(f"Failed: {e}")
        sys.exit(1)