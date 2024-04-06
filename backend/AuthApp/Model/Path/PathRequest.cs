namespace SmartStartBack.Model.Path
{
    public class PathRequest
    {

        public MapPoint[] startEnd { get; set; }
        public bool isSmooth { get; set; }

        public PathRequest(MapPoint[] startEnd, bool isSmooth)
        {
            this.startEnd = startEnd;
            this.isSmooth = isSmooth;
        }

    }
}
