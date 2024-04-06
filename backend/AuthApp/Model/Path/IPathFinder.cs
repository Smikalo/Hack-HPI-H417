namespace SmartStartBack.Model.Path
{
    public interface IPathFinder
    {
        public Task<MapPoint[]> GetShortestPath(MapPoint[] startEnd);

        public MapPoint[] GetSmoothestPath(MapPoint[] startEnd);

    }
}
