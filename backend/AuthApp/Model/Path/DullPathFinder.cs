using AuthApp.Model.Path;
using System.Globalization;
using System.Net.Http;
using System.Text.Json;

namespace SmartStartBack.Model.Path
{
    public class DullPathFinder : IPathFinder

    {
        private HttpClient _httpClient;
        public DullPathFinder(HttpClient httpClient) {
            _httpClient = httpClient;
        }
        public MapPoint[] GetSmoothestPath(MapPoint[] startEnd)
        {
            
            //MapPoint start = new MapPoint(39.96335687073061, -86.13645241042872);
            //MapPoint end = new MapPoint(39.97403352477058, -86.10513864849375);
            MapPoint start = startEnd[0];
            MapPoint end = startEnd[1];
            var readerAm = new StreamReader("D:\\git hub repository\\AuthApp\\Model\\Path\\0e06b441891b68fa8130a04556641b0c1a7f2ad1.json");
            var json = readerAm.ReadToEnd();
            var Info = JsonSerializer.Deserialize<RootobjectSmooth>(json);

            MapPoint[] ans = new MapPoint[Info.elements.Length];
            //ans[0] = start;
            //ans[ans.Length-1] = end;
            for(int i = 0; i < ans.Length; ++i)
            {
                ans[i] = new MapPoint(Info.elements[i].lat, Info.elements[i].lon);
            }

            //HttpRequestMessage messageDisc = new HttpRequestMessage(HttpMethod.Get, "https://router.project-osrm.org/route/v1/car/"+start.lat+","+start.lon+";"+end.lat+","+end.lon+"?overview=full");
            //var responseDisk = await _httpClient.SendAsync(messageDisc);
            // RootobjectFake? infoDisk = JsonSerializer.Deserialize<RootobjectFake>(await responseDisk.Content.ReadAsStringAsync());
            //// MapPoint[] answ = new MapPoint[infoDisk.waypoints.Length];
            // for(int i =0; i < answ.Length; ++i)
            // {
            //     answ[i] = new MapPoint(infoDisk.waypoints[i].location[0], infoDisk.waypoints[i].location[1]);
            // }
            /*
            Random rnd = new Random();
            MapPoint[] answ = new MapPoint[20];
            answ[0] = start;
            answ[19] = end;
            double midLat = (start.lat + end.lat)/2;
            double midlon = (start.lon + end.lon)/2;
            double difLat = Math.Abs(start.lat - end.lat);
            double diflon = Math.Abs(start.lon - end.lon);
            for (int i =1;i< 19; ++i)
            {
                answ[i] = new MapPoint(midLat+(rnd.NextDouble()-1)*difLat, midlon+(rnd.NextDouble()-1)*diflon);
            }
            //return new MapPoint[] { start, new MapPoint(start.lat, end.lon), new MapPoint(end.lat, start.lon), end };*/

            return ans;
        }

        public async Task<MapPoint[]> GetShortestPath(MapPoint[] startEnd)
        {
            //MapPoint start = new MapPoint(39.96335687073061, -86.13645241042872);
            //MapPoint end = new MapPoint(39.97403352477058, -86.10513864849375);
            MapPoint start = startEnd[0];
            MapPoint end = startEnd[1];

            string url = $"https://routing.openstreetmap.de/routed-car/route/v1/driving/{start.lon.ToString(CultureInfo.InvariantCulture)},{start.lat.ToString(CultureInfo.InvariantCulture)};{end.lon.ToString(CultureInfo.InvariantCulture)},{end.lat.ToString(CultureInfo.InvariantCulture)}?overview=false&geometries=polyline&steps=true";
            //Console.WriteLine(url);
            HttpRequestMessage messageDisc = new HttpRequestMessage(HttpMethod.Get, url);
            //HttpRequestMessage messageDisc = new HttpRequestMessage(HttpMethod.Get, "https://routing.openstreetmap.de/routed-car/route/v1/driving/" + start.lat+","+start.lon+";"+end.lat+","+end.lon+"?overview=false&geometries=polyline&steps=true");

            var responseDisk = await _httpClient.SendAsync(messageDisc);
            var answer = await responseDisk.Content.ReadAsStringAsync();
            //Console.WriteLine(answer);
            Rootobject? infoDisk = JsonSerializer.Deserialize<Rootobject>(answer);
            //var infoDisk = JsonSerializer.Deserialize<RootobjectFake>(rootJson);
            int counter = 0;
            foreach (var item in infoDisk.routes) {
                foreach(var root in item.legs)
                {
                    foreach(var step in root.steps)
                    {
                        foreach (var intersection in step.intersections)
                        {
                            ++counter;
                        }
                    }
                }
            }
            MapPoint[] answ = new MapPoint[counter];
            counter = 0;
            foreach (var item in infoDisk.routes)
            {
                foreach (var root in item.legs)
                {
                    foreach (var step in root.steps)
                    {
                        foreach (var intersection in step.intersections)
                        {
                            answ[counter] = new MapPoint(intersection.location[1], intersection.location[0]);
                            ++counter;
                        }
                    }
                }
            }
            /*
            for (int i = 0; i < answ.Length; ++i) { 
                answ[i] = new MapPoint(infoDisk.waypoints[i].location[0], infoDisk.waypoints[i].location[1]);
            }
            
            /*
            Random rnd = new Random();
            MapPoint[] answ = new MapPoint[20];
            answ[0] = start;
            answ[19] = end;
            double midLat = (start.lat + end.lat) / 2;
            double midlon = (start.lon + end.lon) / 2;
            double difLat = Math.Abs(start.lat - end.lat);
            double diflon = Math.Abs(start.lon - end.lon);
            for (int i = 1; i < 19; ++i)
            {
                answ[i] = new MapPoint(midLat + (rnd.NextDouble() - 1) * difLat, midlon + (rnd.NextDouble() - 1) * diflon);
            }
            //return new MapPoint[] { start, new MapPoint(end.lat, start.lon), new MapPoint(start.lat, end.lon), end };*/
            return answ;
        }
    }
}
