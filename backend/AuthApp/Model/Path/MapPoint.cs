using System.Text.Json.Serialization;

namespace SmartStartBack.Model
{
    public class MapPoint
    {
        [JsonPropertyName("lat")]
        public double lat { get; set; }
        [JsonPropertyName("lon")]
        public double lon { get; set; }

        public MapPoint(double lat, double lon) {
            this.lat = lat;
            this.lon = lon;
        }
    }
}
