namespace AuthApp.Model.Path
{


    public class Rootobject
    {
        public string code { get; set; }
        public RouteFake[] routes { get; set; }
        public Waypoint[] waypoints { get; set; }
    }

    public class RouteFake
    {
        public Leg[] legs { get; set; }
        public string weight_name { get; set; }
        public float weight { get; set; }
        public float duration { get; set; }
        public float distance { get; set; }
    }

    public class Leg
    {
        public Step[] steps { get; set; }
        public string summary { get; set; }
        public float weight { get; set; }
        public float duration { get; set; }
        public float distance { get; set; }
    }

    public class Step
    {
        public string geometry { get; set; }
        public Maneuver maneuver { get; set; }
        public string mode { get; set; }
        public string driving_side { get; set; }
        public string name { get; set; }
        public Intersection[] intersections { get; set; }
        public float weight { get; set; }
        public float duration { get; set; }
        public float distance { get; set; }
        public string rotary_name { get; set; }
        public string destinations { get; set; }
        public string _ref { get; set; }
    }

    public class Maneuver
    {
        public int bearing_after { get; set; }
        public int bearing_before { get; set; }
        public float[] location { get; set; }
        public string modifier { get; set; }
        public string type { get; set; }
        public int exit { get; set; }
    }

    public class Intersection
    {
        public int _out { get; set; }
        public bool[] entry { get; set; }
        public int[] bearings { get; set; }
        public float[] location { get; set; }
        public int _in { get; set; }
        public Lane[] lanes { get; set; }
    }

    public class Lane
    {
        public bool valid { get; set; }
        public string[] indications { get; set; }
    }

    public class Waypoint
    {
        public string hint { get; set; }
        public float distance { get; set; }
        public string name { get; set; }
        public float[] location { get; set; }
    }


}
