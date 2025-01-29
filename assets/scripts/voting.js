import { getFirestore, doc, getDoc, setDoc, updateDoc, increment, serverTimestamp } from "https://www.gstatic.com/firebasejs/11.2.0/firebase-firestore.js";

document.addEventListener('DOMContentLoaded', async function() {
    const votingContainer = document.querySelector('.voting-buttons');
    if (!votingContainer) return;
  
    const postId = votingContainer.dataset.postId;
    console.log("Post ID:", postId); // For debugging
    
    const upvoteBtn = votingContainer.querySelector('.upvote');
    const downvoteBtn = votingContainer.querySelector('.downvote');
    const upvoteCount = votingContainer.querySelector('.upvote .upvote-count');
    const downvoteCount = votingContainer.querySelector('.downvote .downvote-count');
  
    // Wait for Firebase to be initialized
    while (!window.firebaseApp) {
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    const db = getFirestore(window.firebaseApp);
    const votesRef = doc(db, 'votes', postId);
    let currentVote = null;
    
    async function initializeVotes() {
        try {
            const docSnap = await getDoc(votesRef);
            
            if (!docSnap.exists()) {
                await setDoc(votesRef, {
                    upvotes: 0,
                    downvotes: 0,
                    createdAt: serverTimestamp()
                });
                upvoteCount.textContent = '0';
                downvoteCount.textContent = '0';
            } else {
                const data = docSnap.data();
                upvoteCount.textContent = data.upvotes || 0;
                downvoteCount.textContent = data.downvotes || 0;
            }
            
            // Check if user has voted before
            currentVote = localStorage.getItem(`voted_${postId}`);
            if (currentVote === 'up') {
                upvoteBtn.classList.add('voted');
                upvoteBtn.querySelector('.vote-icon').classList.add('voted');
            } else if (currentVote === 'down') {
                downvoteBtn.classList.add('voted');
                downvoteBtn.querySelector('.vote-icon').classList.add('voted');
            }
        } catch (error) {
            console.error("Error initializing votes:", error);
            upvoteCount.textContent = '0';
            downvoteCount.textContent = '0';
        }
    }

    await initializeVotes();

    async function updateVoteCounts() {
        const docSnap = await getDoc(votesRef);
        if (docSnap.exists()) {
            const data = docSnap.data();
            upvoteCount.textContent = data.upvotes || 0;
            downvoteCount.textContent = data.downvotes || 0;
        }
    }

    async function handleVote(voteType) {
        try {
            const docSnap = await getDoc(votesRef);
            const updates = {
                lastUpdated: serverTimestamp()
            };

            // Remove previous vote if exists
            if (currentVote) {
                updates[`${currentVote}votes`] = increment(-1);
                const prevBtn = votingContainer.querySelector(`.${currentVote}vote`);
                prevBtn.classList.remove('voted');
                prevBtn.querySelector('.vote-icon').classList.remove('voted');
            }

            // If clicking the same button, just remove the vote
            if (currentVote === voteType) {
                currentVote = null;
                localStorage.removeItem(`voted_${postId}`);
            } else {
                // Add new vote
                updates[`${voteType}votes`] = increment(1);
                const newBtn = votingContainer.querySelector(`.${voteType}vote`);
                newBtn.classList.add('voted');
                newBtn.querySelector('.vote-icon').classList.add('voted');
                currentVote = voteType;
                localStorage.setItem(`voted_${postId}`, voteType);
            }

            if (!docSnap.exists()) {
                await setDoc(votesRef, {
                    upvotes: voteType === 'up' ? 1 : 0,
                    downvotes: voteType === 'down' ? 1 : 0,
                    createdAt: serverTimestamp(),
                    lastUpdated: serverTimestamp()
                });
            } else {
                await updateDoc(votesRef, updates);
            }

            await updateVoteCounts();
        } catch (error) {
            console.error(`Error ${voteType}voting:`, error);
        }
    }
  
    upvoteBtn.addEventListener('click', () => handleVote('up'));
    downvoteBtn.addEventListener('click', () => handleVote('down'));
});