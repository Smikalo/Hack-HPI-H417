﻿namespace AuthApp.Model.Path
{

    public class RootobjectSmooth
    {
        public float version { get; set; }
        public string generator { get; set; }
        public Osm3s osm3s { get; set; }
        public ElementSmooth[] elements { get; set; }
    }

    public class Osm3s
    {
        public DateTime timestamp_osm_base { get; set; }
        public string copyright { get; set; }
    }

    public class ElementSmooth
    {
        public string type { get; set; }
        public long id { get; set; }
        public float lat { get; set; }
        public float lon { get; set; }
        public Tags tags { get; set; }
        public long[] nodes { get; set; }
    }

    public class Tags
    {
        public string TMCcid_58tabcd_1Class { get; set; }
        public string TMCcid_58tabcd_1Direction { get; set; }
        public string TMCcid_58tabcd_1LCLversion { get; set; }
        public string TMCcid_58tabcd_1LocationCode { get; set; }
        public string TMCcid_58tabcd_1NextLocationCode { get; set; }
        public string TMCcid_58tabcd_1PrevLocationCode { get; set; }
        public string crossing { get; set; }
        public string crossingisland { get; set; }
        public string crossingmarkings { get; set; }
        public string crossing_ref { get; set; }
        public string highway { get; set; }
        public string tactile_paving { get; set; }
        public string ele { get; set; }
        public string traffic_signals { get; set; }
        public string traffic_signalsdirection { get; set; }
        public string button_operated { get; set; }
        public string traffic_signalssound { get; set; }
        public string traffic_signalsvibration { get; set; }
        public string kerb { get; set; }
        public string surface { get; set; }
        public string traffic_calming { get; set; }
        public string check_date { get; set; }
        public string access { get; set; }
        public string barrier { get; set; }
        public string bicycle { get; set; }
        public string foot { get; set; }
        public string note { get; set; }
        public string bus { get; set; }
        public string name { get; set; }
        public string public_transport { get; set; }
        public string refBVG { get; set; }
        public string website { get; set; }
        public string gtfsstop_id { get; set; }
        public string image { get; set; }
        public string surveydate { get; set; }
        public string wheelchair { get; set; }
        public string traffic_signalsarrow { get; set; }
        public string noexit { get; set; }
        public string bollard { get; set; }
        public string lit { get; set; }
        public string mapillary { get; set; }
        public string created_by { get; set; }
        public string check_datetactile_paving { get; set; }
        public string maxheight { get; set; }
        public string emergency { get; set; }
        public string motorcar { get; set; }
        public string stopdirection { get; set; }
        public string disusedrefBVG { get; set; }
        public string crossingbicycle { get; set; }
        public string traffic_sign { get; set; }
        public string vehicle { get; set; }
        public string _ref { get; set; }
        public string level { get; set; }
        public string amenity { get; set; }
        public string fee { get; set; }
        public string parking { get; set; }
        public string motor_vehicle { get; set; }
        public string segregated { get; set; }
        public string motorcycle { get; set; }
        public string artist_name { get; set; }
        public string artwork_type { get; set; }
        public string start_date { get; set; }
        public string tourism { get; set; }
        public string direction { get; set; }
        public string crossingkerb_extension { get; set; }
        public string disusedcrossing { get; set; }
        public string entrance { get; set; }
        public string swing_gatetype { get; set; }
        public string local_ref { get; set; }
        public string addrcity { get; set; }
        public string addrcountry { get; set; }
        public string addrhousenumber { get; set; }
        public string addrpostcode { get; set; }
        public string addrstreet { get; set; }
        public string addrsuburb { get; set; }
        public string sourceaddr { get; set; }
        public string contactwebsite { get; set; }
        public string currencyEUR { get; set; }
        public string currencyothers { get; set; }
        public string opening_hours { get; set; }
        public string paymentcash { get; set; }
        public string paymentcoins { get; set; }
        public string paymentcredit_cards { get; set; }
        public string paymentnotes { get; set; }
        public string horse { get; set; }
        public string locked { get; set; }
        public string maxwidth { get; set; }
        public string disusedhighway { get; set; }
        public string disusedbus { get; set; }
        public string disusedpublic_transport { get; set; }
        public string check_datecrossing { get; set; }
        public string gtfs_id { get; set; }
        public string description { get; set; }
        public string display { get; set; }
        public string man_made { get; set; }
        public string monitoringbicycle { get; set; }
        public string monitoringtraffic { get; set; }
        public string _operator { get; set; }
        public string recording { get; set; }
        public string recordingautomated { get; set; }
        public string website2 { get; set; }
        public string website3 { get; set; }
        public string supervised { get; set; }
        public string red_turnright { get; set; }
        public string bicycle_rental { get; set; }
        public string capacity { get; set; }
        public string network { get; set; }
        public string operatortype { get; set; }
        public string crossingbuffer_marking { get; set; }
        public string material { get; set; }
        public string colour { get; set; }
        public string footconditional { get; set; }
        public string lift_gatetype { get; set; }
        public string crossingsignals { get; set; }
        public string cyclewayboth { get; set; }
        public string lanes { get; set; }
        public string maxspeed { get; set; }
        public string nameetymologywikidata { get; set; }
        public string nameetymologywikipedia { get; set; }
        public string oneway { get; set; }
        public string parkingboth { get; set; }
        public string sidewalk { get; set; }
        public string smoothness { get; set; }
        public string wikidata { get; set; }
        public string wikimedia_commons { get; set; }
        public string wikipedia { get; set; }
        public string zonetraffic { get; set; }
        public string cyclewayleft { get; set; }
        public string cyclewayleftoneway { get; set; }
        public string cyclewayright { get; set; }
        public string onewaybicycle { get; set; }
        public string postal_code { get; set; }
        public string sidewalkbothsurface { get; set; }
        public string lit_by_gaslight { get; set; }
        public string cyclewaybothlane { get; set; }
        public string parkingbothorientation { get; set; }
        public string cyclewaysmoothness { get; set; }
        public string cyclewaysurface { get; set; }
        public string lane_markings { get; set; }
        public string maxspeedtype { get; set; }
        public string sidewalkboth { get; set; }
        public string sourcemaxspeed { get; set; }
        public string zonemaxspeed { get; set; }
        public string sidewalkleft { get; set; }
        public string sidewalkright { get; set; }
        public string parkingleft { get; set; }
        public string parkingright { get; set; }
        public string parkingrightorientation { get; set; }
        public string parkingbothreason { get; set; }
        public string service { get; set; }
        public string parkingleftorientation { get; set; }
        public string parkingleftrestriction { get; set; }
        public string cycleway { get; set; }
        public string cyclewaybothsurface { get; set; }
        public string cyclewaybothwidth { get; set; }
        public string sidewalksurface { get; set; }
        public string width { get; set; }
        public string covered { get; set; }
        public string parkingbothfee { get; set; }
        public string dual_carriageway { get; set; }
        public string maxspeedconditional { get; set; }
        public string cyclewayrightlane { get; set; }
        public string parkingrightrestriction { get; set; }
        public string bicycle_road { get; set; }
        public string cyclewayrightwidth { get; set; }
        public string lanespsv { get; set; }
        public string cyclewaybothseparationleft { get; set; }
        public string lanesbackward { get; set; }
        public string lanesforward { get; set; }
        public string maxweight { get; set; }
        public string destination { get; set; }
        public string destinationref { get; set; }
        public string destinationrefto { get; set; }
        public string destinationsymbol { get; set; }
        public string destinationsymbolto { get; set; }
        public string namehe { get; set; }
        public string nameja { get; set; }
        public string namepl { get; set; }
        public string nameru { get; set; }
        public string namezh { get; set; }
        public string shoulder { get; set; }
        public string destinationcolour { get; set; }
        public string destinationto { get; set; }
        public string parkingbothrestriction { get; set; }
        public string nameel { get; set; }
        public string namesr { get; set; }
        public string nameuk { get; set; }
        public string source { get; set; }
        public string junction { get; set; }
        public string turnlanes { get; set; }
        public string placement { get; set; }
        public string transitlanes { get; set; }
        public string cyclewayrightsegregated { get; set; }
        public string sourcewidth { get; set; }
        public string layer { get; set; }
        public string motorroad { get; set; }
        public string tunnel { get; set; }
        public string lanesboth_ways { get; set; }
        public string cyclewayrightsurface { get; set; }
        public string parkingrightrestrictionconditional { get; set; }
        public string bridge { get; set; }
        public string maxweightsigned { get; set; }
        public string noname { get; set; }
        public string bridgename { get; set; }
        public string cyclewaybothsurfacecolour { get; set; }
        public string vehicleconditional { get; set; }
        public string cyclewayrightseparationleft { get; set; }
        public string parkingrightmarkings { get; set; }
        public string parkingrightmarkingstype { get; set; }
        public string hazard { get; set; }
        public string cyclewayleftlane { get; set; }
        public string cyclewayrightsurfacecolour { get; set; }
        public string sidewalkleftsurface { get; set; }
        public string old_name { get; set; }
        public string busway { get; set; }
        public string bicycleforward { get; set; }
        public string psv { get; set; }
        public string parkingleftfee { get; set; }
        public string psvlanes { get; set; }
        public string vehiclelanes { get; set; }
        public string sidewalkrightsurface { get; set; }
        public string cyclewayleftbuffer { get; set; }
        public string cyclewayleftseparationleft { get; set; }
        public string cyclewayleftseparationright { get; set; }
        public string cyclewayleftsurfacecolour { get; set; }
        public string cyclewayleftwidth { get; set; }
        public string overtakingmotorcar { get; set; }
        public string flood_prone { get; set; }
        public string classbicycletouring { get; set; }
        public string parkingrightaccess { get; set; }
        public string parkingrighttaxi { get; set; }
        public string cyclewayrightbicycle { get; set; }
        public string cyclewayrightseparationright { get; set; }
        public string descriptioncovid19 { get; set; }
        public string accesslanes { get; set; }
        public string taxi { get; set; }
        public string buslanes { get; set; }
        public string accessconditional { get; set; }
        public string nat_ref { get; set; }
        public string fixme { get; set; }
        public string cyclewayrightseparationboth { get; set; }
        public string parkingrightfee { get; set; }
        public string cyclewayleftsurface { get; set; }
        public string cyclewayleftbicycle { get; set; }
        public string cyclewaylefttraffic_sign { get; set; }
        public string cyclewayrightbuffer { get; set; }
        public string cyclewayrightoneway { get; set; }
        public string bicyclelanes { get; set; }
        public string cyclewaylanes { get; set; }
        public string buswayright { get; set; }
        public string lanespsvforward { get; set; }
        public string parkingrightrestrictionreason { get; set; }
        public string lanesbus { get; set; }
        public string turnlanesbackward { get; set; }
        public string incline { get; set; }
        public string onewaypsv { get; set; }
        public string cyclewaybothbufferright { get; set; }
        public string cyclewaybothmarkingboth { get; set; }
        public string cyclewaybothseparationright { get; set; }
        public string cyclewaybothtraffic_sign { get; set; }
        public string bus_lane { get; set; }
        public string maxheightsigned { get; set; }
        public string onewaytaxi { get; set; }
        public string check_datecycleway { get; set; }
        public string widthlanes { get; set; }
        public string parkingleftreason { get; set; }
        public string cyclewayrightbufferleft { get; set; }
        public string cyclewayrighttraffic_sign { get; set; }
        public string placementbackward { get; set; }
        public string turnlanesforward { get; set; }
        public string sourcemaxheight { get; set; }
        public string cyclewaylane { get; set; }
        public string construction { get; set; }
        public string constructionend_date { get; set; }
        public string check_dateconstruction { get; set; }
        public string cyclewayleftmarkingleft { get; set; }
        public string hgv { get; set; }
        public string location { get; set; }
        public string parkingleftmarkings { get; set; }
        public string parkingleftmarkingstype { get; set; }
        public string is_incity { get; set; }
    }
}

