#!/usr/bin/perl -w 
#common common
use strict;
use warnings;
use Getopt::Long;
use IO::Socket::INET;
use IO::Select;
eval 'use Time::HiRes "time";';
$SIG{INT}=\&terminate;
$SIG{TERM}=\&terminate;
$SIG{KILL}=\&terminate;
$SIG{CHLD}='IGNORE';
sub show_usage {
	print STDERR "Usage : $0 <-l listening_port> <-r [host:]port> [-f dump_file] [-s speed-limit] [-t expire-time] [-h] [-d]\n";
	print STDERR <<USAGE ;
	-l, --listen-port
		port number to listen on, port under 1024 requires root privilege
	-r, --relay-target
		host and port number to relay , host default set to localhost
	-f, --file-prefix
		file name prefix to dump the output,if exist,append to the current file,default linked to /dev/null
		filename format : prefix.thispid.childid.{recv,send}
	-s, --speed-limit
		limit the forwarding speed .speed is counted in unit of byte
			0 == unlimited
			when speed < 0 , accepts connection and acts like a black hole
			default set to 0 (unlimited)
	-t, --time-to-expire
		limit the live time of a connection, when time expires ,becomes a black hole
			time is counted in seconds,
			0 == never expired
	-d, --debug-level
		debug message output to standard error
	-h, --help
		show you this help
USAGE
	my $message=shift;
	defined $message and print STDERR $message,"\n";
	exit 1;
}

our %LOG=(
	FATAL=>-1,
	NONE=>0,
	WARNING=>1,
	NOTICE=>2,
	TRACE=>3,
	DEBUG=>4,
	LEVEL=>["NONE","WARNING","NOTICE","TRACE","DEBUG","FATAL"],
);
our $debug=$LOG{NONE};
sub writelog {
	my $msglevel=shift;
	defined $msglevel or return undef;
	$msglevel > $debug and return 1;
	my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst)=localtime();
	my $log=sprintf("%s: %02d-%02d %02d:%02d:%02d: %s * %d",@{$LOG{LEVEL}}[$msglevel],$mon+1,$mday+1, $hour,$min,$sec,$0,$$);
	print STDERR "$log @_\n";
	$msglevel < 0 and exit 1;
	return 1;
}

# dealing with parameters and initiation		
our ($showhelp,$listenport,$relaytarget,$relayip,$relayport,$dumpfile,$speed,$expiretime);
GetOptions('help'=>\$showhelp,'debug+'=>\$debug,'listen-port=i'=>\$listenport,'relay-target=s'=>\$relaytarget,
	'file-prefix=s'=>\$dumpfile,'speed-limit=s'=>\$speed,'time-to-expire=s'=>\$expiretime) or show_usage('unknown option');
$showhelp and show_usage();
defined $debug or $debug=$LOG{NONE};
$debug >= scalar @{$LOG{LEVEL}} and $debug=scalar @{$LOG{LEVEL}} - 1;
	
$listenport or show_usage("missing critical parameter -l, program exit.");
$relaytarget or show_usage("missing critical parameter -r, program exit.");
my $relayhostname;
my ($tmphost,$tmpport) =split /:/,$relaytarget,2;
if (defined $tmpport) {
	$relayhostname=$tmphost;
	$relayport=$tmpport;
} else {
	$relayhostname="localhost";
	$relayport=$tmphost;
}
$relayip=hosttoip($relayhostname) or writelog($LOG{FATAL},"$relayhostname can't be resovled.hostname lookup failed");
$relayport=~/\D/ and writelog($LOG{FATAL},"port number must be an positive integer");
($relayport <= 0 || $relayport >= 65536) and writelog($LOG{FATAL},"port number must within (0,65536)");

defined $dumpfile or $dumpfile="";
defined $speed or $speed=0;
defined $expiretime or $expiretime=1800000000;
our $firsttime=time();
writelog($LOG{NOTICE},"relayip=$relayip relayport=$relayport fileprefix=$dumpfile speed limit = $speed expiretime=$expiretime");

# main ()  get down to business
#occupying the file descriptor 3 and 4
our ($nullrecvfd,$nullsendfd);
open $nullrecvfd,">/dev/null" or writelog($LOG{FATAL},"open for 3 recv error,$!");
open $nullsendfd,">/dev/null" or writelog($LOG{FATAL},"open for 4 send error,$!");
my $listensock = IO::Socket::INET->new(LocalPort=>$listenport,Type=>SOCK_STREAM,Reuse=>1, Listen=>SOMAXCONN ) 
	or writelog($LOG{FATAL},"Can't listen on $listenport, error is $!");
writelog($LOG{NOTICE},"set up listen port on $listenport fileno=",fileno($listensock));

my $socks=new IO::Select($listensock);
our %connectedip=();

if (my $timerpid=fork()) {
	for (;;) {
	#block until a new connection from client
		my $new=$listensock->accept();
		if (my $pid=fork()) {
			my (undef,$ip,undef)=split / /,getnameipport($new),3;
			defined $ip or $ip = -1;
			if (exists $connectedip{$ip}) {
				$connectedip{$ip}++;
			} else {
				$connectedip{$ip}=1;
			};
			writelog($LOG{WARNING},$connectedip{$ip}," connections from $ip");
			close $new;
			next;
		} elsif ( !defined $pid) {
			writelog($LOG{WARNING},"forking error.message is $!");
		} else {
		#child process to deal with connection
			close $listensock;
			my $fsock = IO::Socket::INET->new(PeerAddr => $relayip,PeerPort => $relayport, Proto => "tcp" , TYPE => SOCK_STREAM) or writelog($LOG{FATAL},"connect to $relayip:$relayport error,$!");
			if ($dumpfile eq "" or $dumpfile eq "-" ) {
				open(RECV,">&3") or writelog($LOG{FATAL},"dup file descriptor 3 error!");
				open(SEND,">&4") or writelog($LOG{FATAL},"dup file descriptor 3 error!");
			} else {	
				my $dumpname="$dumpfile.".getppid.".$$";
				open (RECV,">$dumpname.recv") or open(RECV,">&$nullrecvfd")
					or writelog($LOG{FATAL},"open $dumpname.recv failed and dup nullfd failed");
				open (SEND,">$dumpname.send") or open(SEND,">&$nullsendfd")
					or writelog($LOG{FATAL},"open $dumpname.send failed and dup nullfd failed");
			}
			writelog($LOG{WARNING}," recv open at ",fileno(RECV)," and send open at ",fileno(SEND));
			my %handles = (
				fileno($new) => {
					sourcesock=>$new,
					targetsock=>$fsock,
					source=>getnameipport($new),
					target=>getnameipport($fsock),
					dumphandle=>*RECV,
					bytesrecv=>0,
					lasttime=>time(),
					expired=>0,
				},
				fileno($fsock) => {
					sourcesock=>$fsock,
					targetsock=>$new,
					source=>getnameipport($fsock),
					target=>getnameipport($new),
					dumphandle=>*SEND,
					bytesrecv=>0,
					lasttime=>time(),
					expired=>0,
				},
			);
			writelog($LOG{NOTICE},"connection source = ",$handles{fileno($new)}{source}," target = ",$handles{fileno($fsock)}{target});
			select $new;$|=1;
			select $fsock;$|=1;
			select RECV;$|=1;
			select SEND;$|=1;
			my $socks=new IO::Select($new,$fsock);
			my $bufsize=($speed <= 0) ? 16384 : $speed;
			writelog($LOG{NOTICE},"speed=$speed ; buffer limit set to $bufsize,first time=$firsttime");
			for (;;) { 
			#loop until connection broken
				my @sread=$socks->can_read(1);
				unless (@sread) {
					for my $key (keys %handles) {
						$handles{$key}{lasttime}=time();
						$handles{$key}{byterecv}=0;
					}
					next;
				};
				foreach my $fd (@sread) {
					my $buffer;
					my $fn=fileno($fd);
					#read data from either side
					recv $fd,$buffer,$bufsize,0;
					writelog($LOG{TRACE},"read  result is ",defined $buffer, " received ",defined $buffer ? length $buffer : -1," bytes");
					unless (defined $buffer) { 
						wrtielog($LOG{WARNING},"recv error from ",$handles{$fn}{source});
						closeallandquit($new,$fsock);
					};
					if ((length $buffer) == 0 ) { 
						writelog($LOG{WARNING},"connection closed by ",$handles{$fn}{source});
						closeallandquit($new,$fsock); 
					};
					#$new是teeport接受的客户端连接，可读说明有数据发送给服务端
					($speed >= 0) or next;
					my $ctime=time();
					my $ifexpire = $firsttime + $expiretime - $ctime ;
					#连接建立时间+设定的过期时间-当前时间 < 0时说明已经过期
					if (($handles{$fn}{expired} == 0) && ($ifexpire  < 0) ) {
						$handles{$fn}{expired}=1;
						$handles{$fn}{targetsock}=$nullsendfd;
						writelog($LOG{WARNING},"connection expired, sending data to null fd");
					};
					writelog($LOG{TRACE},"expire limit is $expiretime, now $ifexpire to go");
					my $success=send $handles{$fn}{targetsock},$buffer,0;
					writelog($LOG{TRACE},"send to ",$handles{$fn}{target}," result is ",defined $success," sent ",defined $success ? $success : -1 ," bytes");
					defined $success or closeallandquit($new,$fsock);
					print {$handles{$fn}{dumphandle}} $buffer;
					$speed == 0 and next;
					$handles{$fn}{bytesrecv}+=$success;
					$handles{$fn}{lasttime} = $ctime -1 if $ctime - $handles{$fn}{lasttime} - 2 >= 0;
					my $waittime= $handles{$fn}{lasttime} + 1 - $ctime;
					#上次发送与当前时间的间隔如果小与1秒并且字节数已经达到限制，需要暂停
					if ( $handles{$fn}{bytesrecv} >= $speed && $waittime >= 0 && $speed != 0) {
						writelog($LOG{DEBUG},"from ",$handles{$fn}{lasttime}," to $ctime wait for $waittime seconds");
						select undef,undef,undef,$waittime;
						$handles{$fn}{lasttime}=time();
						$handles{$fn}{bytesrecv}=0;
					} else {
						$handles{$fn}{bytesrecv}+=$success;
					};	
				}
			}
		}
	}
} elsif (!defined $timerpid) {
	writelog($LOG{FATAL},"forking timer error");
} else {
	($expiretime <= 0) and exit;
	writelog($LOG{WARNING},"expire time is $expiretime ,waiting from $firsttime");
	select undef,undef,undef,$expiretime;
	writelog($LOG{WARNING},"$expiretime seconds passed ,time now ",time()," ",scalar keys %connectedip," connection");
	if ( ($> == 0) && (scalar keys %connectedip != 0) ) {
	#only root(uid=0) can do this
		for my $ip (keys %connectedip) {
			system ("/sbin/iptables -A INPUT -s $ip -j DROP"); 
			writelog($LOG{WARNING},"drop packet from $ip  ,status is $?");
			system ("/sbin/iptables -A OUTPUT -d $ip -j DROP"); 
			writelog($LOG{WARNING},"drop packet to $ip  ,status is $?");
		};
	} else {
		writelog($LOG{WARNING},"your effective uid is $> ,",scalar keys %connectedip," connection");
	};
	exit;
}	
							
sub hosttoip {
	my $hostname=shift;
	defined $hostname or return undef;
	my $ip = gethostbyname($hostname);
	defined $ip or return undef;
	$ip=Socket::inet_ntoa($ip);
	return $ip;
}
sub getnameipport {
	my $s=shift;
	defined $s or return "undef";
	my $sockaddr=getpeername ( $s );
	defined $sockaddr or return "noaddr";
	(my $port,my $addr)=sockaddr_in($sockaddr);
	my $hostname=gethostbyaddr($addr,AF_INET);
	defined $hostname or return "nohostname";
	my $ip=inet_ntoa($addr);
	defined $ip or return "noip";
	return "$hostname $ip $port"; 
}
sub terminate {
	if (defined $expiretime && $> == 0) {
		for my $ip (keys %connectedip) {
			$ip ne "-1" and do {
				system ("/sbin/iptables -D INPUT -s $ip -j DROP"); 
				writelog($LOG{WARNING},"delete drop packet from $ip  ,status is $?");
				system ("/sbin/iptables -D OUTPUT -d $ip -j DROP"); 
				writelog($LOG{WARNING},"delete drop packet to $ip  ,status is $?");
			};
		};
	};
	exit;
}
sub closeallandquit {
	for my $fh (@_) {
		close $fh;
	}
	exit;
}
